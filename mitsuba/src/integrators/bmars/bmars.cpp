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
#include "../path/GitSHA1.h"

#include <OpenImageDenoise/oidn.hpp>

#include <mitsuba/bidir/vertex.h>
#include <mitsuba/bidir/edge.h>

#include <chrono>
#include <sys/stat.h>
#include <unistd.h>

#include "bmars_proc.h"
#include "bmars_aovs.h"

MTS_NAMESPACE_BEGIN

void atomicAdd(Spectrum *dest, const Spectrum &src) {
    for (int c = 0; c < SPECTRUM_SAMPLES; ++c)
        atomicAdd(&(*dest)[c], src[c]);
}

// Copied from EARS implementation
class DenoisingAuxilariesIntegrator : public SamplingIntegrator {
public:
    enum EField {
        EShadingNormal,
        EAlbedo,
    };

    DenoisingAuxilariesIntegrator()
    : SamplingIntegrator(Properties()) {
    }

    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const {
        Spectrum result(0.f);

        if (!rRec.rayIntersect(ray))
            return result;

        Intersection &its = rRec.its;

        switch (m_field) {
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

    std::string toString() const {
        return "DenoisingAuxilariesIntegrator[]";
    }

    EField m_field;
};


class BMARSIntegrator : public Integrator {
public:
    BMARSIntegrator(const Properties &props) : Integrator(props) {
        /* Load the parameters / defaults */
        // Necessary to parse RRSMethod
        std::string rrsStr       = props.getString("rrsStrategy", "classicRRA");
        const int rrDepth        = props.getInteger("rrDepth", 5);
        const Float splittingMax = props.getFloat("splittingMax", 20);
        const Float splittingMin = props.getFloat("splittingMin", 0.05f);
        const bool reusePaths    = props.getBoolean("reusePaths", false);
        // Other configurations
        m_config.splitBSDF          = props.getBoolean("splitBSDF", true);
        m_config.splitNEE           = props.getBoolean("splitNEE", false);
        m_config.splitLP            = props.getBoolean("splitLP", false);
        m_config.disableLP          = props.getBoolean("disableLP", false);
        m_config.lightImage         = props.getBoolean("lightImage", true);
        m_config.budgetAware        = props.getBoolean("budgetAware", false);
        m_config.sfExp              = props.getFloat("sfExp", 1);
        m_config.maxDepth           = props.getInteger("maxDepth", -1);
        m_config.outlierRejection   = props.getInteger("outlierRejection", 10);
        m_config.sampleDirect       = props.getBoolean("sampleDirect", true);
        m_config.showWeighted       = props.getBoolean("showWeighted", false);
        m_config.hideEmitters       = props.getBoolean("hideEmitters", false);
        m_config.strictNormals      = props.getBoolean("strictNormals", true);
        m_saveTrainingFrames        = props.getBoolean("saveTrainFrames", false);
        m_config.shareSF            = props.getBoolean("shareSF", true);
        m_config.shareLP            = props.getBoolean("shareLP", false);
        const long memory_limit     = props.getInteger("memLimit", 24);

        if (!m_config.shareSF && m_config.shareLP) {
            Log(EError, "Cannot share LP splitting factor without sharing splittingfactor in general.");
            exit(-1);
        }
        if (m_config.shareSF && !m_config.shareLP && m_config.splitLP) {
            Log(EError, "Cannot split LP when sharing splitting factors w/o LP share.");
            exit(-1);
        }
        if (m_config.shareSF && m_config.shareLP && !m_config.splitLP) {
            Log(EError, "Need to split LP when trying to share LP.");
            exit(-1);
        }

        /// Parse RRSMethod
        m_config.renderRRSMethod =
                &(m_renderRRSMethod = RRSMethod(splittingMin, splittingMax, rrDepth, rrsStr, reusePaths));

        /// Check and reset flags that are not supported outside of BMARS
        if (m_renderRRSMethod.technique != RRSMethod::EBMARS && m_config.budgetAware) {
            m_config.budgetAware = false;
            Log(EWarn, "Use of splitting factor in MIS, only supported in MARS. Setting to false...");
        }
        if (m_renderRRSMethod.technique != RRSMethod::EBMARS && m_config.splitLP) {
            m_config.splitLP = false;
            Log(EWarn, "Generating multiple light paths, only supported in MARS. Setting to false...");
        }
        if (m_renderRRSMethod.technique != RRSMethod::EBMARS && m_config.splitNEE) {
            m_config.splitNEE = false;
            Log(EWarn, "Generating multiple NEE samples, only supported in MARS. Setting to false...");
        }
        if (m_config.splitLP && reusePaths) {
            Log(EError, "Cannot reuse the same light path and split LP at the same time.");
            exit(-1);
        }

        /// Generating light image from direct light samples only does not make sense
        if (m_renderRRSMethod.maxEmitterDepth() == 1)
            m_config.lightImage = false;

        /// Pointer to structures from outside that the workers need to access
        m_config.imageStatistics  = &m_imageStatistics;
        m_config.cache = &cache;
        m_config.pixelEstimate = &m_pixelEstimate;

        /// Local initializations
        m_oidnDevice = oidn::newDevice();
        m_oidnDevice.commit();

        m_budget = props.getFloat("budget", 30.0f);
        m_imageStatistics.setOutlierRejectionCount(m_config.outlierRejection);

        Log(EInfo, "Setting Octtree memory limit to %d MB", memory_limit * 3);
        cache.setMaximumMemory(long(memory_limit)*1024*1024); /// 24 MiB
        cache.setShareSF(m_config.shareSF);
        cache.setLeafDecay(true);

        #if BDPT_DEBUG == 1
        if (m_config.maxDepth == -1 || m_config.maxDepth > 6) {
            /* Limit the maximum depth when rendering image
               matrices in BDPT debug mode (the number of
               these images grows quadratically with depth) */
            Log(EWarn, "Limiting max. path depth to 6 to avoid an "
                "extremely large number of debug output images");
            m_config.maxDepth = 6;
        }
        #endif

        if (rrDepth <= 0)
            Log(EError, "'rrDepth' must be set to a value greater than zero!");

        if (m_config.maxDepth <= 0 && m_config.maxDepth != -1)
            Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");
    }

    /// Unserialize from a binary data stream
    BMARSIntegrator(Stream *stream, InstanceManager *manager)
     : Integrator(stream, manager) {
        m_config = BMARSConfiguration(stream);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
        m_config.serialize(stream);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
            const RenderJob *job, int sceneResID, int sensorResID,
            int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID,
                sensorResID, samplerResID);

        if (scene->getSubsurfaceIntegrators().size() > 0)
            Log(EError, "Subsurface integrators are not supported "
                "by the bidirectional path tracer!");

        return true;
    }

    void cancel() {
        Scheduler::getInstance()->cancel(m_process);
    }

    void configureSampler(const Scene *scene, Sampler *sampler) {
        /* Prepare the sampler for tile-based rendering */
        sampler->setFilmResolution(scene->getFilm()->getCropSize(), true);
    }

    void updateImageStatistics(Float actualTotalCost) {
        m_imageStatistics.reset(actualTotalCost);
        m_config.imageEarsFactor = m_imageStatistics.earsFactor();
    }

    void updateCaches() {
        if (m_renderRRSMethod.needsTrainingPhase()) {
            cache.build(true);
        }
    }

    void computePixelEstimate(const ref<Film> &film, const Scene *scene) {
        if (!m_renderRRSMethod.needsPixelEstimate()) {
            Log(EDebug, "No need PxEst");
            return;
        }

        /// For bdptRR we need to load a known pixelEstimate to obtain a good efficiency approximation
        if (!m_renderRRSMethod.needsTrainingPhase()) {
            if (!m_pixelEstimate) {
                /// Load accurate pixelEstimate into m_pixelEstimate if desired...
            }
            return;
        }

        const Vector2i size = film->getSize();
        if (!m_pixelEstimate) {
            m_pixelEstimate = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        }
        const int bytePixelStride = m_pixelEstimate->getBytesPerPixel();
        const int byteRowStride = size.x * bytePixelStride;
#ifdef USE_VARWEIGHTED_BITMAP
        if (m_finalImage.hasData())
            m_finalImage.develop(m_pixelEstimate.get());
        else
#endif // USE_VARWEIGHTED_BITMAP
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
    }

    bool renderDenoisingAuxiliaries(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID) {

        Log(EInfo, "Rendering auxiliaries for pixel estimates");

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Properties samplerProperties { "ldsampler" };
        samplerProperties.setInteger("sampleCount", 128);

        ref<Sampler> sampler = static_cast<Sampler *>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), samplerProperties));

        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int samplerResID = sched->registerMultiResource(samplers);
        for (size_t i=0; i<sched->getCoreCount(); ++i)
            samplers[i]->decRef();

        ref<DenoisingAuxilariesIntegrator> integrator = new DenoisingAuxilariesIntegrator();

        bool result = true;

        /// render normals
        film->clear();
        integrator->m_field = DenoisingAuxilariesIntegrator::EField::EShadingNormal;
        result &= integrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_denoiseAuxNormals = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        film->develop(Point2i(0, 0), film->getSize(), Point2i(0, 0), m_denoiseAuxNormals);

        /// render albedo
        film->clear();
        integrator->m_field = DenoisingAuxilariesIntegrator::EField::EAlbedo;
        result &= integrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_denoiseAuxAlbedo = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        film->develop(Point2i(0, 0), film->getSize(), Point2i(0, 0), m_denoiseAuxAlbedo);

        sched->unregisterResource(samplerResID);

        return result;
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    inline bool exists (const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)m_budget, job));
        ref<Scheduler> scheduler = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        ref<Film> film = sensor->getFilm();
        size_t nCores = scheduler->getCoreCount();

        Log(EDebug, "Size of data structures: PathVertex=%i bytes, PathEdge=%i bytes",
            (int) sizeof(PathVertex), (int) sizeof(PathEdge));
        Log(EInfo, "Starting render job (" SIZE_T_FMT " %s, " SSE_STR ") ..",
                                            nCores, nCores == 1 ? "core" : "cores");
        /* Just dump some info to have it available in the exr file */
        Log(EInfo, "Configuration:\n"\
                   "  MARS latest commit hash: %s\n"
                   "  RenderMethod: %s\n"
                   "  splitBSDF:    %s\n"
                   "  splitNEE:     %s\n"
                   "  splitLP:      %s\n"
                   "  budget-aware: %s\n"
                   "  SF exponent:  %.1f\n"
                   "  lightImage:   %s\n"
                   "  reuseLP:      %s\n"
                   "  Max depth:    %d\n"
                   "  RR depth:     %d\n"
                   "  outlierRej:   %d\n"
                   "  Precision:    %s\n"
                   "  AOVs:         %s\n"
                   "  Low Discrep.: %s\n"
                   "  shareSF:      %s\n"
                   "  shareLP:      %s\n"
                   "  disableLP:    %s"
                   ,
                g_GIT_SHA1,
                m_renderRRSMethod.getName().c_str(),
                (m_config.splitBSDF ? "true" : "false"),
                (m_config.splitNEE ? "true" : "false"),
                (m_config.splitLP ? "true" : "false"),
                (m_config.budgetAware ? "true" : "false"),
                m_config.sfExp,
                (m_config.lightImage ? "true" : "false"),
                (m_config.renderRRSMethod->bmarsReuseLP ? "true" : "false"),
                m_config.maxDepth,
                m_renderRRSMethod.rrDepth,
                m_config.outlierRejection,
#ifdef SINGLE_PRECISION
                "SINGLE",
#else // DOUBLE_PRECISION
                "DOUBLE",
#endif
#ifdef MARS_INCLUDE_AOVS
                "enabled",
#else // MARS_INCLUDE_AOVS
                "disabled",
#endif
#ifdef LOW_DISCREPANCY_NUM_SAMPLES
                "enabled",
#else // LOW_DISCREPANCY_NUM_SAMPLES
                "disabled",
#endif
                (m_config.shareSF ? "true" : "false"),
                (m_config.shareLP ? "true" : "false"),
                (m_config.disableLP ? "true" : "false")
                );

        m_config.blockSize = scene->getBlockSize();
        m_config.cropSize = film->getCropSize();
        m_config.imageEarsFactor = INITIAL_EARS_FACTOR;
        m_config.dump();
#ifdef MARS_INCLUDE_AOVS
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

            m_debugFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
            m_debugFilm->addChild(rfilter);
            m_debugFilm->configure();

            m_statsImages.reset(new StatsRecursiveImageBlocks([&]() {
                return new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
            }));
            m_debugImage = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, film->getCropSize(), NULL,
                statsDesc.size * SPECTRUM_SAMPLES + 2
            );
        }

        m_config.statsImages = m_statsImages.get(); // Funny to use smart unique pointer then, ik
#endif

        if (m_renderRRSMethod.needsTrainingPhase())
            renderDenoisingAuxiliaries(scene, queue, job, sceneResID, sensorResID);
        RRSMethod trainMethod = m_renderRRSMethod.technique == RRSMethod::EEARS ?
                                    RRSMethod::Classic() :
                                    RRSMethod::BDPT();

        ref<BMARSProcess> process;
#ifdef USE_VARWEIGHTED_BITMAP
        m_finalImage.clear();
#endif

        /* Check if output path exists, otherwise create the requested directory */
        if (!exists(scene->getDestinationFile().parent_path().string())) {
            if (mkdir(scene->getDestinationFile().parent_path().string().c_str(), 0775) == -1)
                exit(1);
        }

        /* Pre-allocate work results to distribute to workers.
           The reason for this is that the lightImage wr for one iteration should not be cleared.
           This allows us to not add the lightImage on the result every pass, but only once per iteration.
           It is significantly faster as the result is mutex locked as it's not threadsafe to access. */
        BMARSWorkResult *wrs[nCores];
        const auto m_rfilter = film->getReconstructionFilter();
        // for (auto& wr : wrs)
        //     wr = new BMARSWorkResult(m_config, m_rfilter, Vector2i(m_config.blockSize));
        for (int i = 0; i < (int) nCores; ++i)
            wrs[i] = new BMARSWorkResult(m_config, m_rfilter, Vector2i(m_config.blockSize));
        m_config.wrs = wrs;

        // Allocate separate WorkResult for lightImage accumulation
        BMARSWorkResult* liWR = nullptr;
        if (m_config.lightImage)
            liWR = new BMARSWorkResult(m_config, NULL, film->getCropSize());

        Float iterationTime = 1;    // in seconds
        int samplesTaken = 0;       // globally
        Float until = 0;

        auto m_startTime = std::chrono::steady_clock::now();
        for (int i = 0;; ++i) {
            Float timeBeforeIter = computeElapsedSeconds(m_startTime);
            if (timeBeforeIter >= m_budget)
                break;

            m_config.currentRRSMethod = &(m_renderRRSMethod.needsTrainingPhase() && i < TRAIN_ITERATIONS ?
                                   trainMethod : m_renderRRSMethod);

            if (m_config.lightImage) {
                liWR->clearBlock();
                liWR->clearLightImage();
                for (auto& wr : wrs)
                    wr->clearLightImage();
            }
            film->clear();
#ifdef MARS_INCLUDE_AOVS
            m_statsImages->clear();
#endif

            until += iterationTime;
            if (until > m_budget - iterationTime)
                until = m_budget;

            Log(EInfo, "ITERATION %d, until %.1f seconds (%s)", i, until, m_config.currentRRSMethod->getName().c_str());

            size_t spp = 0;
            while (true) {
                m_process = process = new BMARSProcess(job, queue, m_config, liWR);
                process->bindResource("scene", sceneResID);
                process->bindResource("sensor", sensorResID);
                process->bindResource("sampler", samplerResID);

                scheduler->schedule(process);
                scheduler->wait(process);

                spp++;
                m_imageStatistics.applyOutlierRejection();

                const Float progress = computeElapsedSeconds(m_startTime);
                m_progress->update(progress);
                if (progress > until)
                    break;
            }
            samplesTaken += spp;
            const auto elapsed = computeElapsedSeconds(m_startTime);
            Log(EInfo, "  %.2f seconds elapsed, passes this iteration: %d, total passes: %d",
                        elapsed, spp, samplesTaken);

            if (m_config.lightImage) {
                for (auto& wr : wrs)
                    liWR->putLightImage(wr);
                process->develop(spp);
            }

            /// Update octtree
            updateCaches();
            updateImageStatistics(elapsed - timeBeforeIter);
#ifdef USE_VARWEIGHTED_BITMAP
            const bool hasVarianceEstimate = i > 0 || !m_renderRRSMethod.needsPixelEstimate();
            m_finalImage.add(
                film, spp,
                m_renderRRSMethod.performsInvVarWeighting() ?
                    (hasVarianceEstimate ? m_imageStatistics.squareError().average() : 0) :
                    1
            );
#endif // USE_VARWEIGHTED_BITMAP
            computePixelEstimate(film, scene); // Update pixel estimates for next iteration

            if (m_saveTrainingFrames) {
                const auto size = film->getSize();
                ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat32, size);
                m_finalImage.develop(bitmap.get());

                fs::path path = scene->getDestinationFile();
                path = path.parent_path() / (path.leaf().string() + "__train-" + formatString("%03d", i) + ".exr");
                Log(EInfo, "Saving training frame to %s", path.c_str());
                bitmap->write(path);
            }

            #if BDPT_DEBUG == 1
                fs::path path = scene->getDestinationFile();
                if (m_config.lightImage)
                    process->getResult()->dump(m_config, path.parent_path(), path.stem());
            #endif


            if ((i % M_DOUBLE_TIME) == (M_DOUBLE_TIME - 1)) {
                /// double the sample count of a render pass every M_DOUBLE_TIME passes.
                iterationTime *= 2;
            }
        }

        const auto renderTime = timeString(queue->getRenderTime(job), true);


#ifdef MARS_INCLUDE_AOVS
        Vector2i size = film->getSize();
        ref<Bitmap> image = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        film->develop(Point2i(0, 0), size, Point2i(0, 0), image);

        auto statsBitmaps = m_statsImages->getBitmaps();
        Float* debugImage = m_debugImage->getBitmap()->getFloatData();

        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x) {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);

                /// write out debug channels
                for (int i = 0; i < SPECTRUM_SAMPLES; ++i) *(debugImage++) = pixel[i];

                for (auto &b : statsBitmaps) {
                    Spectrum v = b->getPixel(pos);
                    for (int i = 0; i < SPECTRUM_SAMPLES; ++i) *(debugImage++) = v[i];
                }

                *(debugImage++) = 1.0f;
                *(debugImage++) = 1.0f;
            }

        m_debugFilm->setBitmap(m_debugImage->getBitmap());

        {
            /// output debug image
            std::string suffix = "-dbg-BMARS";
            fs::path destPath = scene->getDestinationFile();
            fs::path debugPath = destPath.parent_path() / (
                destPath.leaf().string()
                + suffix
                + ".exr"
            );

            m_debugFilm->setDestinationFile(debugPath, 0);
            m_debugFilm->develop(scene, 0.0f);
        }
#endif

#ifdef USE_VARWEIGHTED_BITMAP
        ref<Bitmap> finalBitmap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        m_finalImage.develop(finalBitmap.get());
        film->setBitmap(finalBitmap);
#endif // USE_VARWEIGHTED_BITMAP

        m_progress = nullptr;
        return process->getReturnStatus() == ParallelProcess::ESuccess;
    }

    MTS_DECLARE_CLASS()
private:
    RRSMethod m_renderRRSMethod;

    mutable Octtree cache;
    mutable ImageStatistics m_imageStatistics;

    std::unique_ptr<StatsRecursiveImageBlocks> m_statsImages;
    mutable ref<ImageBlock> m_debugImage;
    mutable ref<Film> m_debugFilm;

    oidn::DeviceRef m_oidnDevice;
    ref<Bitmap> m_pixelEstimate;
    ref<Bitmap> m_denoiseAuxNormals;
    ref<Bitmap> m_denoiseAuxAlbedo;

    mutable std::unique_ptr<ProgressReporter> m_progress;

#ifdef USE_VARWEIGHTED_BITMAP
    struct WeightedBitmapAccumulator {
        void clear() {
            m_scrap = nullptr;
            m_bitmap = nullptr;
            m_spp = 0;
            m_weight = 0;
        }

        bool hasData() const {
            return m_weight > 0;
        }

        void add(const ref<Film> &film, int spp, Float avgVariance = 1) {
            if (avgVariance == 0 && m_weight > 0) {
                SLog(EError, "Cannot add an image with unknown variance to an already populated accumulator");
                return;
            }

            const Vector2i size = film->getSize();
            const long floatCount = size.x * size.y * long(SPECTRUM_SAMPLES);

            if (!m_scrap) {
                m_scrap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
            }
            film->develop(Point2i(0, 0), size, Point2i(0, 0), m_scrap);

            ///

            if (!m_bitmap) {
                m_bitmap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);

                float *m_bitmapData = m_bitmap->getFloat32Data();
                for (long i = 0; i < floatCount; ++i) {
                    m_bitmapData[i] = 0;
                }
            }

            float *m_bitmapData = m_bitmap->getFloat32Data();
            if (avgVariance > 0 && m_weight == 0 && m_spp > 0) {
                /// reweight previous frames that had unknown variance with our current variance estimate
                const Float reweight = 1 / avgVariance;
                for (long i = 0; i < floatCount; ++i) {
                    m_bitmapData[i] *= reweight;
                }
                m_weight += m_spp * reweight;
            }

            const Float weight = avgVariance > 0 ? spp / avgVariance : spp;
            const float *m_scrapData = m_scrap->getFloat32Data();
            for (long i = 0; i < floatCount; ++i) {
                m_bitmapData[i] += m_scrapData[i] * weight;
            }

            m_weight += avgVariance > 0 ? weight : 0;
            m_spp += spp;
        }

        void develop(Bitmap *dest) const {
            if (!m_bitmap) {
                SLog(EWarn, "Cannot develop bitmap, as no data is available");
                return;
            }

            const Vector2i size = m_bitmap->getSize();
            const long floatCount = size.x * size.y * long(SPECTRUM_SAMPLES);

            const Float weight = m_weight == 0 ? m_spp : m_weight;
            float *m_destData = dest->getFloat32Data();
            const float *m_bitmapData = m_bitmap->getFloat32Data();
            for (long i = 0; i < floatCount; ++i) {
                m_destData[i] = weight > 0 ? m_bitmapData[i] / weight : 0;
            }
        }

        void develop(const fs::path &path) const {
            if (!m_scrap) {
                SLog(EWarn, "Cannot develop bitmap, as no data is available");
                return;
            }

            develop(m_scrap.get());
            m_scrap->write(path);
        }

    private:
        mutable ref<Bitmap> m_scrap;
        ref<Bitmap> m_bitmap;
        Float m_weight;
        int m_spp;
    } m_finalImage;
#endif // USE_VARWEIGHTED_BITMAP

    ref<ParallelProcess> m_process;
    BMARSConfiguration m_config;

    Float m_budget;
    bool m_saveTrainingFrames = true;
};

MTS_IMPLEMENT_CLASS_S(BMARSIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(BMARSIntegrator, "BMARS - Bidirectional MARS");
MTS_NAMESPACE_END
