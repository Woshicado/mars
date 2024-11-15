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

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/bidir/util.h>
#include "effmis_proc.h"

MTS_NAMESPACE_BEGIN

thread_local StatsRecursiveImageBlockCache *StatsRecursiveImageBlockCache::instance = nullptr;
thread_local StatsRecursiveDescriptorCache *StatsRecursiveDescriptorCache::instance = nullptr;
thread_local StatsRecursiveValuesCache *StatsRecursiveValuesCache::instance = nullptr;

static StatsCounter sTotalCameraPathLength("EffMIS", "Total camera path length", ENumberValue);
static StatsCounter sTotalLightPathLength("EffMIS", "Total light path length", ENumberValue);

constexpr int StatsPickedK = 5;

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class EffMISRenderer : public WorkProcessor
{
public:
    EffMISRenderer(const EffMISContext* config) : m_context(config) {}

    EffMISRenderer(Stream *stream, InstanceManager *manager)
        : WorkProcessor(stream, manager), m_context(nullptr) {
        Assert(true); // Not supported
    }

    virtual ~EffMISRenderer() {}

    void serialize(Stream *stream, InstanceManager *manager) const
    {
        context().serialize(stream);
    }

    ref<WorkUnit> createWorkUnit() const
    {
        return new RectangularWorkUnit();
    }

    ref<WorkResult> createWorkResult(int workerIDX) const
    {
        /// With original mitsuba, we only have access to workerIDX by parsing the thread name.
        /// However, the caller knows the workerIndex, so we can just pass it.
        // const char* threadName = &(Thread::getThread()->getName().c_str()[3]);
        // const int id = atoi(threadName);
        // Assert(id == workerIDX);
        return context().wrs[workerIDX];
    }

    void prepare()
    {
        Scene *scene = static_cast<Scene *>(getResource("scene"));
        m_scene = new Scene(scene);
        m_sampler = static_cast<Sampler *>(getResource("sampler"));
        m_sensor = static_cast<Sensor *>(getResource("sensor"));
        m_rfilter = m_sensor->getFilm()->getReconstructionFilter();
        m_scene->removeSensor(scene->getSensor());
        m_scene->addSensor(m_sensor);
        m_scene->setSensor(m_sensor);
        m_scene->setSampler(m_sampler);
        m_scene->wakeup(NULL, m_resources);
        m_scene->initializeBidirectional();
    }

    struct RoundedFloat {
        Float Original;
        int Pick;
    };

    inline static RoundedFloat stochasticRound(Float primary, Float value) {
        int num = (int)value;
        Float residual = value - num;
        if (primary < residual) num++;
        return RoundedFloat { value, num };
    }

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop)
    {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
        EffMISWorkResult *result = static_cast<EffMISWorkResult *>(workResult);
        bool needsTimeSample = m_sensor->needsTimeSample();
        Float time = m_sensor->getShutterOpen();

        result->setOffset(rect->getOffset());
        result->setSize(rect->getSize());
        result->clearBlock();
        m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

#ifdef EFFMIS_INCLUDE_AOVS
        const auto &block = result->getImageBlock();
        static thread_local StatsRecursiveImageBlocks blocks([&]()
                                                             {
            auto b = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
            return b; });

        for (auto &b : blocks.blocks)
        {
            b->setOffset(block->getOffset());
            b->clear();
        }
#endif

        StatsRecursiveValues stats;

#if defined(MTS_DEBUG_FP)
        enableFPExceptions();
#endif
        // Temporary emitter/sensor subpath used for NEE, emissive hits and LT
        // Valid, unless methods rely on length
        Path tmpSubpath(2);
        tmpSubpath.append(nullptr, nullptr);
        tmpSubpath.append(nullptr, nullptr);

        Path sensorSubpath;
        Path* emitterSubpaths = (Path*)alloca(sizeof(Path) * context().numConnections);
        for(size_t c = 0; c < context().numConnections; ++c)
            new(&emitterSubpaths[c]) Path(); // Placement-new

        /* Determine the necessary random walk depths based on properties of
           the endpoints */
        int sensorDepth = context().config.maxDepth;
        int emitterDepth = context().config.maxDepth;

        /* Go one extra step if there are emitters that can be intersected */
        if (!m_scene->hasDegenerateEmitters() && sensorDepth != -1)
            ++sensorDepth;
        if (!m_scene->hasDegenerateSensor() && emitterDepth != -1)
            ++emitterDepth;

        for (size_t i = 0; i < m_hilbertCurve.getPointCount(); ++i)
        {
            Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);

            for (size_t j = 0; j < context().internalIterations; j++)
            {
                stats.reset();
                if (stop)
                    break;

                if (needsTimeSample)
                    time = m_sensor->sampleTime(m_sampler->next1D());

                /* Start new emitter and sensor subpaths */
                sensorSubpath.initialize(m_scene, time, ERadiance, m_pool);
                sensorSubpath.randomWalkFromPixel(m_scene, m_sampler, sensorDepth, offset, context().config.rrDepth, m_pool);

                sTotalCameraPathLength += sensorSubpath.length();
                sTotalCameraPathLength.incrementBase();

                const Float numConnectionsF = context().numberOfConnections(offset);
                const RoundedFloat numConnections = stochasticRound(m_sampler->next1D(), numConnectionsF);

                for (int c = 0; c < numConnections.Pick; ++c) {
                    emitterSubpaths[c].initialize(m_scene, time, EImportance, m_pool);
                    emitterSubpaths[c].randomWalk(m_scene, m_sampler, emitterDepth, context().config.rrDepth, EImportance, m_pool);

                    sTotalLightPathLength += emitterSubpaths[c].length();
                    sTotalLightPathLength.incrementBase();
                }

#ifdef EFFMIS_INCLUDE_AOVS
                const Point2 samplePos = sensorSubpath.vertex(1)->getSamplePosition();
#endif

                evaluate(result, numConnections, sensorSubpath, tmpSubpath, emitterSubpaths, stats);

                sensorSubpath.release(m_pool);
                for (int c = 0; c < numConnections.Pick; ++c)
                    emitterSubpaths[c].release(m_pool);

                m_sampler->advance();

#ifdef EFFMIS_INCLUDE_AOVS
                stats.put(blocks, samplePos, /* alpha */ 1.0);
#endif
            }
        }

#ifdef EFFMIS_INCLUDE_AOVS
        context().statsImages->put(blocks);
#endif

#if defined(MTS_DEBUG_FP)
        disableFPExceptions();
#endif

        /* Make sure that there were no memory leaks */
        Assert(m_pool.unused());
    }

    /// Evaluate the contributions of the given eye and light paths
    void evaluate(EffMISWorkResult *wr, const RoundedFloat& numConnections, Path &sensorSubpath, Path& tmpEmitterSubpath, Path* emitterSubpaths, StatsRecursiveValues &stats) {
        Point2 samplePos = sensorSubpath.vertex(1)->getSamplePosition();

        /* Compute the combined weights along the two subpaths */
        Spectrum *radianceWeights = (Spectrum *)alloca(sensorSubpath.vertexCount() * sizeof(Spectrum));
        BidirUtils::computeCachedWeights(sensorSubpath, radianceWeights, ERadiance);

        Spectrum **importanceWeights = (Spectrum **)alloca(numConnections.Pick * sizeof(Spectrum **));
        for (int c = 0; c < numConnections.Pick; ++c) {
            importanceWeights[c] = (Spectrum*) alloca(emitterSubpaths[c].vertexCount() * sizeof(Spectrum));
            BidirUtils::computeCachedWeights(emitterSubpaths[c], importanceWeights[c], EImportance);
        }

        const int maxN     = context().config.maxDepth + 3;
        Float* pdfImp      = (Float *) alloca(maxN * sizeof(Float)),
             * pdfRad      = (Float *) alloca(maxN * sizeof(Float));
        bool * connectable = (bool *)  alloca(maxN * sizeof(bool)),
             * isNull      = (bool *)  alloca(maxN * sizeof(bool));
        BidirPdfs pathPdfs = BidirPdfs::create(maxN, pdfRad, pdfImp, connectable, isNull);

        Spectrum contrib(0.0f);
        int maxT = (int)sensorSubpath.vertexCount() - 1;

#ifdef EFFMIS_INCLUDE_AOVS
        int pathCount = 0;
        int totalPathLengthCount = 0;

        if (maxT > 2)
        {
            const PathVertex* vt = sensorSubpath.vertex(2);
            if (!vt->isDegenerate() && vt->isSurfaceInteraction()) {
                const BSDF *bsdf = vt->getIntersection().getBSDF();
                const Spectrum albedo = bsdf->getDiffuseReflectance(vt->getIntersection()) + bsdf->getSpecularReflectance(vt->getIntersection());
                stats.albedo.add(albedo);
            }
        }

        stats.numConDensity.add(numConnections.Original);
        stats.numConPick.add(numConnections.Pick);
#endif

        for (int t = 2 /* Skip super node and sensor */; t <= maxT; ++t)
        {
            const PathVertex* vt = sensorSubpath.vertex(t);
            if (vt->isDegenerate())
                continue;

            if (vt->isSurfaceInteraction() && vt->getIntersection().isEmitter()) {
                contrib += handleEmissionHits(wr, samplePos, sensorSubpath, t, radianceWeights, numConnections.Original, stats, tmpEmitterSubpath, pathPdfs);

#ifdef EFFMIS_INCLUDE_AOVS
                ++pathCount;
                totalPathLengthCount += sensorSubpath.length();
#endif
            }

            if (context().config.sampleDirect && t + 1 <= context().config.maxDepth) {
                contrib += handleNEE(wr, samplePos, sensorSubpath, t, radianceWeights, numConnections.Original, stats, tmpEmitterSubpath, pathPdfs);

#ifdef EFFMIS_INCLUDE_AOVS
                ++pathCount;
                totalPathLengthCount += sensorSubpath.length() + 1;
#endif
            }

            for (int c = 0; c < numConnections.Pick; ++c) {
                contrib += handleConnection(wr, samplePos, sensorSubpath, emitterSubpaths[c], t, radianceWeights, importanceWeights[c], numConnections.Original, stats, pathPdfs);

#ifdef EFFMIS_INCLUDE_AOVS
                const int bidirStart = context().config.sampleDirect ? 2 : 1;
                const int sl = std::max(0, emitterSubpaths[c].length() - bidirStart);
                pathCount += sl;
                totalPathLengthCount += sl * sensorSubpath.length() + sl + sl * (bidirStart + emitterSubpaths[c].length()) / 2; // Basic arithmetic series
#endif
            }
        }

        // TODO: t <= 1 (lightImage)

#ifdef EFFMIS_INCLUDE_AOVS
        if (pathCount > 0)
        {
            stats.avgPathLength.add(totalPathLengthCount / (Float)pathCount);
            stats.numPaths.add(pathCount);
        }
#endif

        wr->putSample(samplePos, contrib);
    }

    // (t >= 2 && s == 1)
    Spectrum handleNEE(EffMISWorkResult *wr, const Point2& pixel, const Path &sensorSubpath, int t,
        const Spectrum* radianceWeights, Float connectionDensity, StatsRecursiveValues &stats,
        Path& tmpEmitterSubpath, BidirPdfs& pathPdfs) {
        PathVertex vsPred, vs;
        PathEdge vsEdge, connectionEdge;

        PathVertex* vt = sensorSubpath.vertex(t);
        PathVertex* vtPred = sensorSubpath.vertex(t - 1);
        PathEdge * vtEdge = sensorSubpath.edge(t - 1);

        Spectrum value = radianceWeights[t] * vt->sampleDirect(m_scene, m_sampler, &vsPred, &vsEdge, &vs, EImportance);
        if (value.isZero())
            return value;

        value *= vt->eval(m_scene, vtPred, &vs, ERadiance);
        if (value.isZero())
            return value;

        RestoreMeasureHelper rmh(vt);
        vt->measure = EArea;

        int interactions = context().config.maxDepth - t;
        if (!connectionEdge.pathConnectAndCollapse(m_scene, &vsEdge, &vs, vt, vtEdge, interactions))
            return Spectrum(0.0f);

        /* Account for the terms of the measurement contribution that are coupled to the connection edge */
        value *= connectionEdge.evalCached(&vs, vt, PathEdge::ETransmittance | PathEdge::ECosineRad);
        if (value.isZero())
            return value;

        Assert(value.isValid());

        // MIS
        tmpEmitterSubpath.clear();
        tmpEmitterSubpath.append(&vsPred);
        tmpEmitterSubpath.append(&vsEdge, &vs);
        pathPdfs.gather(m_scene, tmpEmitterSubpath, &connectionEdge, sensorSubpath, 1, t, context().config.sampleDirect, context().config.lightImage);
        const Float misWeight = pathPdfs.computeMIS(1, t, context().config.sampleDirect, context().config.lightImage, connectionDensity);
        Assert(std::isfinite(misWeight) && misWeight >= 0 && misWeight <= 1);

        // Moments
        if (context().acquireMoments) {
            const auto proxy = pathPdfs.computeProxyWeights(t+2, context().config.sampleDirect, context().config.lightImage);
            Assert(proxy.PathTracing > 0);
#ifdef EFFMIS_INCLUDE_AOVS
            if (t+2 == StatsPickedK) {
                stats.proxyPTMIS.add(proxy.PathTracing);
                stats.proxyConMIS.add(proxy.Connection);
            }
#endif
            context().putMomentSample(wr, pixel, value * misWeight, proxy);
        }

#if EFFMIS_BDPT_DEBUG == 1
        wr->putDebugSample(1, t, pixel, value, misWeight);
#endif

        return value * misWeight;
    }

    // (t >= 2 && s == 0)
    Spectrum handleEmissionHits(EffMISWorkResult *wr, const Point2& pixel, const Path &sensorSubpath, int t,
        const Spectrum* radianceWeights, Float connectionDensity, StatsRecursiveValues &stats,
        Path& tmpEmitterSubpath, BidirPdfs& pathPdfs) {
        PathVertex* vt = sensorSubpath.vertex(t);
        // PathVertex* vtPred = sensorSubpath.vertex(t - 1);
        PathEdge* vtEdge = sensorSubpath.edge(t - 1);

        Spectrum value = radianceWeights[t] * vt->getIntersection().Le(vtEdge->d);

        if (value.isZero() || !vt->cast(m_scene, PathVertex::EEmitterSample))
            return Spectrum(0.0f);

        Assert(value.isValid());

        // MIS
        tmpEmitterSubpath.clear();
        PathVertex endpoint;
        endpoint.makeEndpoint(m_scene, sensorSubpath.vertex(0)->getTime(), EImportance);
        tmpEmitterSubpath.append(&endpoint);
        pathPdfs.gather(m_scene, tmpEmitterSubpath, vtEdge, sensorSubpath, 0, t, context().config.sampleDirect, context().config.lightImage);
        const Float misWeight = pathPdfs.computeMIS(0, t, context().config.sampleDirect, context().config.lightImage, connectionDensity);
        Assert(std::isfinite(misWeight) && misWeight >= 0 && misWeight <= 1);

        // Moments
        if (context().acquireMoments) {
            const auto proxy = pathPdfs.computeProxyWeights(t+1, context().config.sampleDirect, context().config.lightImage);
            Assert(proxy.PathTracing > 0);
#ifdef EFFMIS_INCLUDE_AOVS
            if (t+1 == StatsPickedK) {
                stats.proxyPTMIS.add(proxy.PathTracing);
                stats.proxyConMIS.add(proxy.Connection);
            }
#endif
            context().putMomentSample(wr, pixel, value * misWeight, proxy);
        }

#if EFFMIS_BDPT_DEBUG == 1
        wr->putDebugSample(0, t, pixel, value, misWeight);
#endif

        return value * misWeight;
    }

    // (t >= 2 && s >= 2)
    Spectrum handleConnection(EffMISWorkResult *wr, const Point2& pixel, Path &sensorSubpath, Path& emitterSubpath, int t,
        const Spectrum* radianceWeights, const Spectrum* importanceWeights, Float connectionDensity, StatsRecursiveValues &stats,
        BidirPdfs& pathPdfs) {
        PathEdge connectionEdge;

        PathVertex *vtPred = sensorSubpath.vertexOrNull(t - 1);
        PathVertex *vt     = sensorSubpath.vertex(t);
        PathEdge* vtEdge   = sensorSubpath.edgeOrNull(t - 1);

        /* Determine the range of sensor vertices to be traversed,
            while respecting the specified maximum path length */
        int maxS = (int)emitterSubpath.vertexCount() - 1;
        if (context().config.maxDepth != -1)
            maxS = std::min(maxS, context().config.maxDepth + 1 - t);

        const int bidirStart = context().config.sampleDirect ? 2 : 1;

        Spectrum contrib = Spectrum(0.0f);
        for (int s = maxS; s >= bidirStart; --s) {
            PathVertex* vsPred = emitterSubpath.vertexOrNull(s - 1);
            PathVertex* vs     = emitterSubpath.vertex(s);
            PathEdge* vsEdge   = emitterSubpath.edgeOrNull(s - 1);

            /* Can't connect degenerate endpoints */
            if (!vs->isConnectable() || !vt->isConnectable()) {
                continue;
            }

            /* Temporarily force vertex measure to EArea. Needed to
                handle BSDFs with diffuse + specular components */
            RestoreMeasureHelper rmh0(vs), rmh1(vt);

            Spectrum value = importanceWeights[s] * radianceWeights[t] *
                        vs->eval(m_scene, vsPred, vt, EImportance) *
                        vt->eval(m_scene, vtPred, vs, ERadiance) / connectionDensity;

            if (value.isZero())
                continue;

            vs->measure = vt->measure = EArea;

            int remaining = context().config.maxDepth - s - t + 1;
            Assert(remaining >= 0);
            if (!connectionEdge.pathConnectAndCollapse(m_scene, vsEdge, vs, vt, vtEdge, remaining))
                continue;

            /* Account for the terms of the measurement contribution
                function that are coupled to the connection edge */
            value *= connectionEdge.evalCached(vs, vt, PathEdge::EGeneralizedGeometricTerm);
            if (value.isZero())
                continue;

            Assert(value.isValid());

            // MIS
            pathPdfs.gather(m_scene, emitterSubpath, &connectionEdge, sensorSubpath, s, t, context().config.sampleDirect, context().config.lightImage);
            const Float misWeight = pathPdfs.computeMIS(s, t, context().config.sampleDirect, context().config.lightImage, connectionDensity);
            Assert(std::isfinite(misWeight) && misWeight >= 0 && misWeight <= 1);

            // Moments
            if (context().acquireMoments) {
                const auto proxy = pathPdfs.computeProxyWeights(s+t+1, context().config.sampleDirect, context().config.lightImage);
                Assert(proxy.Connection > 0);
#ifdef EFFMIS_INCLUDE_AOVS
            if (s+t+1 == StatsPickedK) {
                stats.proxyPTMIS.add(proxy.PathTracing);
                stats.proxyConMIS.add(proxy.Connection);
            }
#endif
                context().putMomentSample(wr, pixel, value * misWeight, proxy);
            }

#if EFFMIS_BDPT_DEBUG == 1
            wr->putDebugSample(s, t, pixel, value, misWeight);
#endif

            contrib += value * misWeight;
        }

        return contrib;
    }

    ref<WorkProcessor> clone() const
    {
        return new EffMISRenderer(m_context);
    }

    [[nodiscard]] inline const EffMISContext& context() const { return *m_context; }

    MTS_DECLARE_CLASS()
private:
    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    mutable ref<Sampler> m_sampler;
    mutable MemoryPool m_pool;
    ref<ReconstructionFilter> m_rfilter;
    const EffMISContext* m_context;
    HilbertCurve2D<uint8_t> m_hilbertCurve;
};

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

EffMISProcess::EffMISProcess(const RenderJob *parent, RenderQueue *queue, const EffMISContext &ctx, EffMISWorkResult* wr)
    : BlockedRenderProcess(parent, queue, ctx.blockSize)
    , m_context(ctx)
{
    /// Progress is tracked from the outside.
    this->disableProgress();
    m_result = wr;
    Assert(!m_context.config.lightImage || wr);

    sTotalCameraPathLength.reset();
    sTotalLightPathLength.reset();
}

ref<WorkProcessor> EffMISProcess::createWorkProcessor() const
{
    return new EffMISRenderer(&m_context);
}

void EffMISProcess::developLightImage(int spp)
{
    if (!m_context.config.lightImage)
        return;

    LockGuard lock(m_resultMutex);
    const ImageBlock *lightImage = m_result->getLightImage();
    m_film->setBitmap(m_result->getImageBlock()->getBitmap());
    m_film->addBitmap(lightImage->getBitmap(), 1.0f / spp);
    m_queue->signalRefresh(m_parent);
}

void EffMISProcess::processResult(const WorkResult *wr, bool cancelled)
{
    if (cancelled)
        return;

    const EffMISWorkResult *result = static_cast<const EffMISWorkResult *>(wr);
    ImageBlock *block = const_cast<ImageBlock *>(result->getImageBlock());

    LockGuard lock(m_resultMutex);
    if(m_context.config.lightImage)
        m_result->putBlock(result);
    else
        m_film->put(block);
    m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);
}

Float EffMISProcess::getAverageCameraPathLength() const
{
    if (sTotalCameraPathLength.getBase() == 0)
        return 0;
    return sTotalCameraPathLength.getValue() / (Float) sTotalCameraPathLength.getBase();
}

Float EffMISProcess::getAverageLightPathLength() const
{
    if (sTotalLightPathLength.getBase() == 0)
        return 0;
    return sTotalLightPathLength.getValue() / (Float) sTotalLightPathLength.getBase();
}

MTS_IMPLEMENT_CLASS_S(EffMISRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(EffMISProcess, false, BlockedRenderProcess)
MTS_NAMESPACE_END
