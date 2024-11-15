/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "effmis.h"
#include "../path/GitSHA1.h"

#ifndef EFFMIS_NO_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

#include <mitsuba/bidir/vertex.h>
#include <mitsuba/bidir/edge.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/statistics.h>

#include <chrono>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>

#include "effmis_aovs.h"
#include "effmis_proc.h"
#include "costs.h"
#include "optimizer.h"

MTS_NAMESPACE_BEGIN

void atomicAdd(Spectrum *dest, const Spectrum &src)
{
    for (int c = 0; c < SPECTRUM_SAMPLES; ++c)
        atomicAdd(&(*dest)[c], src[c]);
}

void EffMISContext::setupPTCandidates() {
    connectionCandidates = {0, 1};
}

void EffMISContext::setupBidirCandidates() {
    connectionCandidates = {0, 1, 2, 4, 8};
}

void EffMISContext::putMomentSample(EffMISWorkResult* res, const Point2 &samplePos, const Spectrum &weight, const BidirPdfs::ProxyWeights &proxyWeights) const
{
    // We compute the second moment of the average value across all color channels.
    Float w = weight.average();

    // Outlier rejection
    if (config.usePixelEstimate && config.outlierFactor > 1) {
        const Float p = pixelEstimate ? pixelEstimate->get()->getPixel(Point2i(std::floor(samplePos.x), std::floor(samplePos.y))).average() : Float(0.5);
        w = std::min(w, p * config.outlierFactor);
    }

    SAssert(w >= 0);
    const Float w2 = w * w;

    // Precompute terms where possible
    const Float con = proxyWeights.Connection / ProxyNumConnections;
    const Float curCon = numberOfConnections(samplePos);

    SAssert(std::isfinite(w2));

    // Update the second moment estimates of all candidates.
    size_t index = 0;
    for (const int candidateC: connectionCandidates) {
        // Proxy weights multiplied by pilot sample count, divided by proxy sample count
        const Float a = proxyWeights.PathTracing + con * curCon;

        // Proxy weights multiplied by candidate sample count, divided by proxy count
        const Float b = proxyWeights.PathTracing + con * candidateC;

        const Float correctionFactor = a / b;

        SAssert(std::isfinite(correctionFactor));
        SAssert(correctionFactor > 0);

        res->putMomentSample(index, samplePos, correctionFactor * w2);
        ++index;
    }
}

// Copied from EARS implementation
class DenoisingAuxilariesIntegrator : public SamplingIntegrator
{
public:
    enum EField
    {
        EShadingNormal,
        EAlbedo,
    };

    DenoisingAuxilariesIntegrator()
        : SamplingIntegrator(Properties())
    {
    }

    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const
    {
        Spectrum result(0.f);

        if (!rRec.rayIntersect(ray))
            return result;

        Intersection &its = rRec.its;

        switch (m_field)
        {
        case EShadingNormal:
            result.fromLinearRGB(its.shFrame.n.x, its.shFrame.n.y, its.shFrame.n.z);
            break;
        case EAlbedo:
            result = its.shape->getBSDF()->getDiffuseReflectance(its);
            break;
        default:
            Log(EError, "Internal error!");
        }

        return result;
    }

    std::string toString() const
    {
        return "DenoisingAuxilariesIntegrator[]";
    }

    EField m_field;
};

class EffMISIntegrator : public Integrator
{
public:
    EffMISIntegrator(const Properties &props) : Integrator(props)
    {
        /* Load the parameters / defaults */
        m_context.config.maxDepth = props.getInteger("maxDepth", -1);
        m_context.config.rrDepth = props.getInteger("rrDepth", 5);
        m_context.config.lightImage = props.getBoolean("lightImage", false); /* True version NOT TESTED! */
        m_context.config.sampleDirect = props.getBoolean("sampleDirect", true);
        m_context.config.numConnections = props.getInteger("connections", 1);
        m_context.config.pilotIterations = props.getInteger("pilotIterations", 1);
        m_context.config.perPixelConnections = props.getBoolean("perPixelConnections", true); // Default false for paper
        m_context.config.usePixelEstimate = props.getBoolean("usePixelEstimate", true);
        m_context.config.useBidirPilot = props.getBoolean("useBidirPilot", true);
        m_context.config.outlierFactor = props.getFloat("outlierFactor", 50); // 50x larger will be clamped by default

#ifdef EFFMIS_NO_OIDN
        if (m_context.config.usePixelEstimate)
            Log(EError, "Trying to use pixel estimates, but OIDN is disabled");
#endif

        /// Local initializations
#ifndef EFFMIS_NO_OIDN
        m_oidnDevice = oidn::newDevice();
        m_oidnDevice.commit();
#endif

        m_budget = props.getFloat("budget", 30.0f);

#if EFFMIS_BDPT_DEBUG == 1
        if (m_context.config.maxDepth == -1 || m_context.config.maxDepth > 6)
        {
            /* Limit the maximum depth when rendering image
               matrices in BDPT debug mode (the number of
               these images grows quadratically with depth) */
            Log(EWarn, "Limiting max. path depth to 6 to avoid an "
                       "extremely large number of debug output images");
            m_context.config.maxDepth = 6;
        }
#endif
        if (m_context.config.lightImage)
            Log(EError, "Setting 'lightImage' to true is not tested!");

        if (m_context.config.rrDepth <= 0)
            Log(EError, "'rrDepth' must be set to a value greater than zero!");

        if (m_context.config.maxDepth <= 0 && m_context.config.maxDepth != -1)
            Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");

        // Initialize
        m_context.setupBidirCandidates();
    }

    /// Unserialize from a binary data stream
    EffMISIntegrator(Stream *stream, InstanceManager *manager)
        : Integrator(stream, manager)
    {
        m_context = EffMISContext(stream);
    }

    void serialize(Stream *stream, InstanceManager *manager) const
    {
        Integrator::serialize(stream, manager);
        m_context.serialize(stream);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int sensorResID,
                    int samplerResID)
    {
        Integrator::preprocess(scene, queue, job, sceneResID,
                               sensorResID, samplerResID);

        if (scene->getSubsurfaceIntegrators().size() > 0)
            Log(EError, "Subsurface integrators are not supported "
                        "by the bidirectional path tracer!");

        return true;
    }

    void cancel()
    {
        if (m_process)
            Scheduler::getInstance()->cancel(m_process);
    }

    void configureSampler(const Scene *scene, Sampler *sampler)
    {
        /* Prepare the sampler for tile-based rendering */
        sampler->setFilmResolution(scene->getFilm()->getCropSize(), true);
    }

    inline void clearLightImages() {
        m_context.wr->clearLightImage();
        for (size_t i = 0; i < m_context.nCores; ++i)
            m_context.wrs[i]->clearLightImage();
    }

    inline void clearAll(Film* film) {
        m_context.wr->clearBlock();
#if EFFMIS_BDPT_DEBUG == 1
        m_context.wr->clearDebugBlocks();
#endif
        for (size_t i = 0; i < m_context.nCores; ++i) {
            m_context.wrs[i]->clearBlock();
#if EFFMIS_BDPT_DEBUG == 1
            m_context.wrs[i]->clearDebugBlocks();
#endif
        }

        clearLightImages();

        film->clear();
#ifdef EFFMIS_INCLUDE_AOVS
        m_statsImages->clear();
#endif
    }

    inline void clearMoments() {
        m_context.wr->clearMoments();

        for (size_t i = 0; i < m_context.nCores; ++i)
            m_context.wrs[i]->clearMoments();
    }

    inline void consumeMoments(int spp) {
        for (size_t i = 0; i < m_context.nCores; ++i)
            m_context.wr->putMomentBlocks(m_context.wrs[i]);

        if (spp > 1)
            m_context.wr->scaleMomentBlocks(1 / (Float)spp);
    }

    inline void consumeLightImage(int spp) {
        if (!m_context.config.lightImage)
            return;

        m_context.wr->clearLightImage();
        for (size_t i = 0; i < m_context.nCores; ++i)
            m_context.wr->putLightImage(m_context.wrs[i]);

        m_process->developLightImage(spp);
    }

    float computePixelEstimate(int spp, ref<Film> &film, const Scene *scene) {
#ifndef EFFMIS_NO_OIDN
        if (!m_context.config.usePixelEstimate)
            return 0;

        consumeLightImage(spp);

        const auto denoiseTime = std::chrono::steady_clock::now();
        const Vector2i size = film->getSize();
        if (!m_pixelEstimate)
            m_pixelEstimate = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);

        const int bytePixelStride = m_pixelEstimate->getBytesPerPixel();
        const int byteRowStride = size.x * bytePixelStride;
        film->develop(Point2i(0, 0), size, Point2i(0, 0), m_pixelEstimate);

        auto filter = m_oidnDevice.newFilter("RT");
        filter.setImage("color", m_pixelEstimate->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.setImage("albedo", m_denoiseAuxAlbedo->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.setImage("normal", m_denoiseAuxNormals->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.setImage("output", m_pixelEstimate->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.set("hdr", true);
        filter.commit();
        filter.execute();

        const char *error;
        if (m_oidnDevice.getError(error) != oidn::Error::None) {
            Log(EError, "OpenImageDenoise: %s", error);
        } else {
            Log(EInfo, "OpenImageDenoise finished successfully");
        }

#ifdef EFFMIS_INCLUDE_AOVS
        fs::path destPath = scene->getDestinationFile();
        m_pixelEstimate->write(destPath.parent_path() / "effmis_pixelestimate.exr");
#endif

        m_context.pixelEstimate = &m_pixelEstimate;

        return computeElapsedSeconds(denoiseTime);
#else // EFFMIS_NO_OIDN
        return 0;
#endif
    }

    void renderDenoisingAuxiliaries(Scene *scene, RenderQueue *queue, const RenderJob *job,
                                    int sceneResID, int sensorResID)
    {
#ifndef EFFMIS_NO_OIDN
        if (!m_context.config.usePixelEstimate)
            return;

        Log(EInfo, "Rendering auxiliaries for pixel estimates");

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Properties samplerProperties{"ldsampler"};
        samplerProperties.setInteger("sampleCount", 128);

        ref<Sampler> sampler = static_cast<Sampler *>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), samplerProperties));

        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i = 0; i < sched->getCoreCount(); ++i)
        {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int samplerResID = sched->registerMultiResource(samplers);
        for (size_t i = 0; i < sched->getCoreCount(); ++i)
            samplers[i]->decRef();

        ref<DenoisingAuxilariesIntegrator> integrator = new DenoisingAuxilariesIntegrator();

        /// render normals
        film->clear();
        integrator->m_field = DenoisingAuxilariesIntegrator::EField::EShadingNormal;
        integrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_denoiseAuxNormals = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        film->develop(Point2i(0, 0), film->getSize(), Point2i(0, 0), m_denoiseAuxNormals);

        /// render albedo
        film->clear();
        integrator->m_field = DenoisingAuxilariesIntegrator::EField::EAlbedo;
        integrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_denoiseAuxAlbedo = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        film->develop(Point2i(0, 0), film->getSize(), Point2i(0, 0), m_denoiseAuxAlbedo);

        sched->unregisterResource(samplerResID);
#endif
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start)
    {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    inline bool exists(const std::string &name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    bool renderStep(RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> scheduler = Scheduler::getInstance();

        m_process = new EffMISProcess(job, queue, m_context, m_context.wr);
        m_process->bindResource("scene", sceneResID);
        m_process->bindResource("sensor", sensorResID);
        m_process->bindResource("sampler", samplerResID);

        scheduler->schedule(m_process);
        scheduler->wait(m_process);

        if (m_process->getReturnStatus() != ParallelProcess::ESuccess) {
            Log(EError, "Camera pass did not finish properly");
            return false;
        }

        return true;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID)
    {
        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)m_budget, job));

        ref<Scheduler> scheduler = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        ref<Film> film = sensor->getFilm();
        size_t sampleCount = scene->getSampler()->getSampleCount();
        size_t nCores = scheduler->getCoreCount();

        if (m_context.config.pilotIterations == 0)
            m_context.config.usePixelEstimate = false;

        if (!exists(scene->getDestinationFile().parent_path().string()))
        {
            if (mkdir(scene->getDestinationFile().parent_path().string().c_str(), 0775) == -1)
                exit(1);
        }

        Log(EDebug, "Size of data structures: PathVertex=%i bytes, PathEdge=%i bytes",
            (int)sizeof(PathVertex), (int)sizeof(PathEdge));

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " samples, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, nCores, nCores == 1 ? "core" : "cores");

        /* Just dump some info to have it available in the exr file */
        Log(EInfo, "Configuration:\n"
                   "  MARS latest commit hash: %s\n"
                   "  lightImage:   %s\n"
                   "  Max depth:    %d\n"
                   "  Initial connections:    %d\n",
            g_GIT_SHA1,
            (m_context.config.lightImage ? "true" : "false"),
            m_context.config.maxDepth,
            m_context.config.numConnections);

        m_context.blockSize = scene->getBlockSize();
        m_context.cropSize = film->getCropSize();
        m_context.sampleCount = sampleCount;

        m_context.dump();

#ifdef EFFMIS_INCLUDE_AOVS
        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);

        {
            /// debug film with additional channels
            StatsRecursiveDescriptor statsDesc;

            auto properties = Properties(film->getProperties());
            properties.setString("pixelFormat", statsDesc.types);
            properties.setString("channelNames", statsDesc.names);
            std::cout << properties.toString() << std::endl;
            auto rfilter = film->getReconstructionFilter();

            m_debugFilm = static_cast<Film *>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
            m_debugFilm->addChild(rfilter);
            m_debugFilm->configure();

            m_statsImages.reset(new StatsRecursiveImageBlocks([&]()
                                                              { return new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize()); }));
            m_debugImage = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, film->getCropSize(), NULL,
                                          statsDesc.size * SPECTRUM_SAMPLES + 2);
        }

        m_context.statsImages = m_statsImages.get(); // Funny to use smart unique pointer then, ik
#endif

        // First setup the larger block of candidates
        if (m_context.config.pilotIterations > 0 && m_context.config.perPixelConnections)
            m_context.numConnectionsImage = new Bitmap(Bitmap::EPixelFormat::ELuminance, Bitmap::EComponentFormat::EFloat, film->getSize());

        renderDenoisingAuxiliaries(scene, queue, job, sceneResID, sensorResID);

        if (m_context.config.usePixelEstimate && m_context.config.outlierFactor > 1)
            Log(EInfo, "Using outlier rejection for moments");

        EffMISWorkResult *wrs[nCores];
        m_context.nCores = nCores;

        const auto m_rfilter = film->getReconstructionFilter();
        for (int i = 0; i < (int)nCores; ++i) {
            wrs[i] = new EffMISWorkResult(i, m_context, m_rfilter, Vector2i(m_context.blockSize));
            wrs[i]->incRef();
        }
        m_context.wrs = wrs;
        m_context.wr = new EffMISWorkResult(-1, m_context, nullptr, m_context.cropSize);
        m_context.wr->incRef();

        Float denoiseTimeSeconds = 0;
        Float optimizerTimeSeconds = 0;
        const auto startTime = std::chrono::steady_clock::now();

        clearAll(film);

        size_t spp = 0;
        if(m_context.config.pilotIterations > 0) {
            {
                if (m_context.config.useBidirPilot) {
                    // Use PT candidates and switch to bidir later
                    m_context.setupPTCandidates();
                }

                Log(EInfo, "Starting Pilot PT");
                clearMoments();

                m_context.acquireMoments = true;
                m_context.numConnections = 0;
                m_context.internalIterations = m_context.config.pilotIterations;

                Log(EInfo, "Using technique [Connections=%i, LT=%s, NEE=%s]",
                    m_context.numConnections,
                    m_context.config.lightImage ? "true" : "false",
                    m_context.config.sampleDirect ? "true" : "false");
                if (!renderStep(queue, job, sceneResID, sensorResID, samplerResID))
                    return false;

                spp += m_context.internalIterations;
                m_progress->update(computeElapsedSeconds(startTime));
                m_context.acquireMoments = false;

                consumeMoments(spp);

                // Compute pixel estimate
                denoiseTimeSeconds += computePixelEstimate(spp, film, scene);

                m_context.avgCameraPathLength = m_process->getAverageCameraPathLength();
                m_context.avgLightPathLength = m_process->getAverageLightPathLength(); // Zero

                const auto optTime = std::chrono::steady_clock::now();
                EffMISCosts costs;
                costs.update(m_context.numberOfPixels(), m_process->getAverageCameraPathLength(), m_process->getAverageLightPathLength());

                Log(EInfo, "Applying filter on computed moments");
                EffMISOptimizer::setup(m_context, m_context.wr);

                Log(EInfo, "Optimizing with avg. path length %.2f", costs.AverageCameraPathLength);
                const int candidate = EffMISOptimizer::optimizeGlobal(m_context, m_context.wr, costs, m_pixelEstimate.get());

                optimizerTimeSeconds += computeElapsedSeconds(optTime);

                m_context.numConnections = candidate;
            }

            if (m_context.numConnections > 0) {
                // We used PT in a Bidir scene, assuming the results are bad and full of outliers. Get rid of it.
                Log(EInfo, "Clearing previous results");
                clearAll(film);
                spp=0;
            }

            if (m_context.numConnections > 0 && m_context.config.useBidirPilot) {
                m_context.setupBidirCandidates();

                Log(EInfo, "Starting Pilot BDPT");
                clearMoments();

                m_context.acquireMoments = true;
                m_context.internalIterations = m_context.config.pilotIterations;

                Log(EInfo, "Using technique [Connections=%i, LT=%s, NEE=%s]",
                    m_context.numConnections,
                    m_context.config.lightImage ? "true" : "false",
                    m_context.config.sampleDirect ? "true" : "false");
                if (!renderStep(queue, job, sceneResID, sensorResID, samplerResID))
                    return false;

                spp += m_context.internalIterations;
                m_progress->update(computeElapsedSeconds(startTime));
                m_context.acquireMoments = false;

                consumeMoments(spp);

                // Compute pixel estimate
                denoiseTimeSeconds += computePixelEstimate(spp, film, scene);

                m_context.avgCameraPathLength = m_process->getAverageCameraPathLength();
                m_context.avgLightPathLength = m_process->getAverageLightPathLength();

                const auto optTime = std::chrono::steady_clock::now();
                EffMISCosts costs;
                costs.update(m_context.numberOfPixels(), m_process->getAverageCameraPathLength(), m_process->getAverageLightPathLength());

                Log(EInfo, "Applying filter on computed moments");
                EffMISOptimizer::setup(m_context, m_context.wr);

                Log(EInfo, "Optimizing with avg. path length %.2f (camera) and %.2f (light) ", costs.AverageCameraPathLength, costs.AverageLightPathLength);
                int candidate = EffMISOptimizer::optimizeGlobal(m_context, m_context.wr, costs, m_pixelEstimate.get());
                if (candidate > 0 && m_context.config.perPixelConnections) {
                    Log(EInfo, "Optimizing number of connections per pixel");
                    candidate = EffMISOptimizer::optimizePerPixel(m_context, m_context.wr, costs, m_pixelEstimate.get());
                    m_context.usePerPixelConnections = true;
                }

                optimizerTimeSeconds += computeElapsedSeconds(optTime);

                m_context.numConnections = candidate;
            }
        } else {
            // Keep user selection
            m_context.numConnections = m_context.config.numConnections;
        }

        Log(EInfo, "Using technique [Connections=%i, LT=%s, NEE=%s]",
            m_context.numConnections,
            m_context.config.lightImage ? "true" : "false",
            m_context.config.sampleDirect ? "true" : "false");

        m_context.internalIterations = 1; // No iterations inside the process
        m_progress->update(computeElapsedSeconds(startTime));
        while (true)
        {
            if (!renderStep(queue, job, sceneResID, sensorResID, samplerResID))
                return false;
            spp += m_context.internalIterations;

            const Float progress = computeElapsedSeconds(startTime);
            m_progress->update(progress);
            if (progress > m_budget)
                break;
        }
        const auto elapsed = computeElapsedSeconds(startTime);
        Log(EInfo, "  %.2f seconds elapsed, spp: %d", elapsed, spp);
        Log(EInfo, "  Average path length: %f (Camera), %f (Light)", m_process->getAverageCameraPathLength(), m_process->getAverageLightPathLength());
        Log(EInfo, "  %.2f seconds elapsed optimizing", optimizerTimeSeconds);
        if (m_context.config.usePixelEstimate)
            Log(EInfo, "  %.2f seconds elapsed denoising for pixel estimate", denoiseTimeSeconds);

        consumeLightImage(spp);

#if EFFMIS_BDPT_DEBUG == 1
        {
            for (auto &wr : cwrs)
                m_context.cwr->putDebugBlock(wr);
            m_context.cwr->checkMIS(m_context, 1.0f / spp);
            const fs::path path = scene->getDestinationFile();
            m_context.cwr->dumpDebug(scene, m_context, path.parent_path(), path.stem(), 1.0f / spp);
        }
#endif

#ifdef EFFMIS_INCLUDE_AOVS
        Vector2i size = film->getSize();
        ref<Bitmap> image = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        film->develop(Point2i(0, 0), size, Point2i(0, 0), image);

        auto statsBitmaps = m_statsImages->getBitmaps();
        Float *debugImage = m_debugImage->getBitmap()->getFloatData();

        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x)
            {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);

                /// write out debug channels
                for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                    *(debugImage++) = pixel[i];

                for (auto &b : statsBitmaps)
                {
                    Spectrum v = b->getPixel(pos);
                    for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                        *(debugImage++) = v[i];
                }

                *(debugImage++) = 1.0f;
                *(debugImage++) = 1.0f;
            }

        m_debugFilm->setBitmap(m_debugImage->getBitmap());

        {
            /// output debug image
            std::string suffix = "-dbg-EffMIS";
            fs::path destPath = scene->getDestinationFile();
            fs::path debugPath = destPath.parent_path() / (destPath.leaf().string() + suffix + ".exr");

            m_debugFilm->setDestinationFile(debugPath, 0);
            m_debugFilm->develop(scene, 0.0f);

            if (m_context.config.pilotIterations > 0)
                m_context.wr->dumpMoments(scene, m_context, destPath.parent_path(), destPath.leaf().string() + "_candidates", 1.0f);
        }

#endif

        m_progress = nullptr;
        return true;
    }

    MTS_DECLARE_CLASS()
private:
    ref<EffMISProcess> m_process;
    EffMISContext m_context;
    Float m_budget;

    mutable std::unique_ptr<ProgressReporter> m_progress;

    std::unique_ptr<StatsRecursiveImageBlocks> m_statsImages;
    mutable ref<ImageBlock> m_debugImage;
    mutable ref<Film> m_debugFilm;

    ref<Bitmap> m_pixelEstimate;

#ifndef EFFMIS_NO_OIDN
    oidn::DeviceRef m_oidnDevice;
    ref<Bitmap> m_denoiseAuxNormals;
    ref<Bitmap> m_denoiseAuxAlbedo;
#endif
};

MTS_IMPLEMENT_CLASS_S(EffMISIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(EffMISIntegrator, "Efficiency aware bidirectional path tracer");
MTS_NAMESPACE_END
