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

#include "bmars_proc.h"
#include "bmars_aovs.h"

MTS_NAMESPACE_BEGIN

thread_local StatsRecursiveImageBlockCache *StatsRecursiveImageBlockCache::instance = nullptr;
thread_local StatsRecursiveDescriptorCache *StatsRecursiveDescriptorCache::instance = nullptr;
thread_local StatsRecursiveValuesCache *StatsRecursiveValuesCache::instance = nullptr;

inline bool FLT_EQ(Float val1, Float val2, Float tolerance=5e-4f) {
    return (val1 - val2) < tolerance &&
           (val2 - val1) < tolerance;
}

inline bool VEC_EQ(Vector val1, Vector val2, Float tolerance=5e-4f) {
    SAssert(val1.dim == val2.dim);

    bool eq = true;
    for (int i = 0; i < val1.dim; i++)
        eq = eq && FLT_EQ(val1[i], val2[i], tolerance);
    return eq;
}

inline bool SPEC_EQ(Spectrum val1, Spectrum val2, Float tolerance=5e-4f) {
    SAssert(val1.dim == val2.dim);

    bool eq = true;
    for (int i = 0; i < val1.dim; i++)
        eq = eq && FLT_EQ(val1[i], val2[i], tolerance);
    return eq;
}


struct SplitMapEntry {
    SplitMapEntry()
        : LR(std::numeric_limits<Float>::quiet_NaN()),
        NEE(std::numeric_limits<Float>::quiet_NaN()),
        LP(std::numeric_limits<Float>::quiet_NaN()) {};
    SplitMapEntry(Float All)
        : LR(All), NEE(All), LP(All) {};
    SplitMapEntry(Float LR, Float NEE, Float LP)
        : LR(LR), NEE(NEE), LP(LP) {};
    SplitMapEntry(const SplitMapEntry& other)
        : SplitMapEntry(other.LR, other.NEE, other.LP) {};
    SplitMapEntry(SplitMapEntry&& other)
        : LR(std::move(other.LR)), NEE(std::move(other.NEE)), LP(std::move(other.LP)) {};

    inline bool epsEQ(const SplitMapEntry &other, Float eps=5e-4f) const {
        return FLT_EQ(LR, other.LR, eps) &&
            FLT_EQ(LP, other.LP, eps) &&
            FLT_EQ(NEE, other.NEE, eps);
    }
    inline void operator=(const SplitMapEntry &other) {
        std::tie(LR, NEE, LP) = other.as_tuple();
    }
    inline std::tuple<Float, Float, Float> as_tuple() const {
        return { LR, NEE, LP };
    }
    // Returns true if no value is nan and none exceeds the valid range
    // By default valid min/max are neg. inf and inf, respectively
    inline bool isValid(Float valid_min= -1 * std::numeric_limits<Float>::infinity(),
                Float valid_max=std::numeric_limits<Float>::infinity(),
                int depth=0, int rrDepth=0, bool sBSDF=true, bool sLP=false, bool sNEE=false,
                bool bsdfHasSmoothComponent=false) const {

        if (!(!(std::isnan(LR) || std::isnan(LP) || std::isnan(NEE))
                && ((  ((valid_min <= LR  && LR  <= valid_max) || (!sBSDF && LR <= 1)            || (depth <= rrDepth        && LR  == 1))
                    && ((valid_min <= NEE && NEE <= valid_max) || (!sNEE && NEE == 1)            || (!bsdfHasSmoothComponent && NEE == 0))
                    && ((valid_min <= LP  && LP  <= valid_max) || (!sLP && (LP == 1 || LP == 0)) || (!bsdfHasSmoothComponent && LP  == 0)))
                || (depth <= rrDepth && LR == 1 && NEE == 1 && (LP == 1 || LP == 0)))))
            SLog(EInfo, "%f\t%f\t%f", LR, LP, NEE);
        return !(std::isnan(LR) || std::isnan(LP) || std::isnan(NEE))
                && ((  ((valid_min <= LR  && LR  <= valid_max) || (!sBSDF && LR <= 1)  || (depth <= rrDepth        && LR  == 1))
                    && ((valid_min <= NEE && NEE <= valid_max) || (!sNEE && NEE == 1)  || (!bsdfHasSmoothComponent && NEE == 0))
                    && ((valid_min <= LP  && LP  <= valid_max) || (LP == 1 || LP == 0) || (!bsdfHasSmoothComponent && LP  == 0)))
                || (depth <= rrDepth && LR == 1 && NEE == 1 && (LP == 1 || LP == 0)));
    }
    inline Float getMultiplier() const {
        return LR * NEE * LP;
    }

    Float LR;
    Float NEE;
    Float LP;
};

using SplitMap = std::vector<SplitMapEntry>;
// using SplitMapPtr = SplitMap*;

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class BMARSRenderer : public WorkProcessor {
public:
    /// the cost of ray tracing + direct illumination sample (in seconds)
    static constexpr Float COST_NEE  = 0.3e-7 * 1.7;

    /// the cost of ray tracing + BSDF/camera sample (in seconds)
    static constexpr Float COST_BSDF = 0.3e-7 * 1.7;

private:
    /// Storing this for the splitting factor calculation
    Spectrum m_pixelEstimate { 0.5f };

    mutable SplitMap splitFactors;

    enum BDPTConnection {
        SELF_ONLY    = 0,
        NOSELF       = 1,
        NOSELF_NONEE = 2
    };

    int m_emitterDepth;

    struct LiOutput {
        Spectrum reflected { 0.f };
        Spectrum emitted { 0.f };
        Float cost { 0.f };

        int numSamples { 0 };
        int neeSamples { 0 };
        int lpSamples { 0 };
        Float depthAcc { 0.f };
        Float depthWeight { 0.f };

        void markAsLeaf(int depth) {
            depthAcc = depth;
            depthWeight = 1;
        }

        Float averagePathLength() const {
            return depthWeight > 0 ? depthAcc / depthWeight : 0;
        }

        Float numberOfPaths() const {
            return depthWeight;
        }

        Spectrum totalContribution() const {
            return reflected + emitted;
        }
    };

public:
    BMARSRenderer(const BMARSConfiguration &config) : m_config(config) { }

    BMARSRenderer(Stream *stream, InstanceManager *manager)
        : WorkProcessor(stream, manager), m_config(stream) { }

    virtual ~BMARSRenderer() { }

    void serialize(Stream *stream, InstanceManager *manager) const {
        m_config.serialize(stream);
    }

    ref<WorkUnit> createWorkUnit() const {
        return new RectangularWorkUnit();
    }

    ref<WorkResult> createWorkResult(int workerIDX) const {
        /// With original mitsuba, we only have access to workerIDX by parsing the thread name.
        /// However, the caller knows the workerIndex, so we can just pass it.
        /// We want to do it like this, s.t. it is allowed that a workResult might not have been
        /// reset between subsequente calls. E.g., relevant for lightImage performance!
        // const char* threadName = &(Thread::getThread()->getName().c_str()[3]);
        // const int id = atoi(threadName);
        // Assert(id == workerIDX);
        return m_config.wrs[workerIDX];
    }

    void prepare() {
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

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
        BMARSWorkResult *result = static_cast<BMARSWorkResult *>(workResult);
        bool needsTimeSample = m_sensor->needsTimeSample();
        Float time = m_sensor->getShutterOpen();

        result->setOffset(rect->getOffset());
        result->setSize(rect->getSize());
        result->clearBlock();
        m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

#ifdef MARS_INCLUDE_AOVS
        const auto &block = result->getImageBlock();
        static thread_local StatsRecursiveImageBlocks blocks([&]() {
            auto b = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
            return b;
        });

        for (auto &b : blocks.blocks) {
            b->setOffset(block->getOffset());
            b->clear();
        }
#endif
        StatsRecursiveValues stats;

        #if defined(MTS_DEBUG_FP)
            enableFPExceptions();
        #endif

        Path emitterPath;
        /* Use emitterPath only for NEE to imitate EARS implementation if requested
          Go one extra step if the sensor can be intersected */
        m_emitterDepth = ! m_config.disableLP ?
                     std::min(m_config.renderRRSMethod->maxEmitterDepth(),
                                m_config.maxDepth + ((int) (!m_scene->hasDegenerateSensor() && m_config.maxDepth != -1)))
                     : 1;

        // EARS variables
        static thread_local OutlierRejectedAverage blockStatistics;
        blockStatistics.resize(m_config.outlierRejection);
        if (m_config.imageStatistics->hasOutlierLowerBound()) {
            blockStatistics.setRemoteOutlierLowerBound(m_config.imageStatistics->outlierLowerBound());
        }

        Float depthAcc = 0;
        Float depthWeight = 0;
        Float primarySplit = 0;
        Float samplesTaken = 0;
        Float neeSplit = 0;
        Float lpSplit = 0;

        /* Data structure to cache the splitting factors of the used vertices in the paths */
        for (size_t i=0; i<m_hilbertCurve.getPointCount(); ++i) {
            Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);
            Assert(splitFactors.empty());

            // I am requiring a sampler with an arbitrary number of samples here, e.g., independent sampler
            constexpr int sppPerPass = 1;
            for (size_t j = 0; j < sppPerPass; j++) {
                stats.reset();
                if (stop)
                    break;

                /* Update pre-cached pixel estimate */
                if (m_config.pixelEstimate->get())
                    m_pixelEstimate = (*m_config.pixelEstimate)->getPixel(offset);

                if (needsTimeSample)
                    time = m_sampler->next1D();

                /* Start new emitter subpaths */
                emitterPath.initialize(m_scene, time, EImportance, m_pool);

                /// We only need to trace this if reused for all connections.
                /// Remember to remove costs further down if this is changed
                /* Perform a random walk to sample a light path */
                if (m_config.currentRRSMethod->bmarsReuseLP)
                    emitterPath.randomWalk(m_scene, m_sampler, m_emitterDepth, m_config.currentRRSMethod->rrDepth,
                                        ETransportMode::EImportance, m_pool);

                /* Perform EARS routine and connect every generated vertex to all light vertices */
                LiOutput output; Point2 samplePos;
                std::tie(samplePos, output) = EARSWalkAndEvaluate(
                        emitterPath, offset, time, result, blockStatistics, stats);

                emitterPath.release(m_pool);
                m_sampler->advance();

#ifdef MARS_INCLUDE_AOVS
                stats.put(blocks, samplePos, /* alpha */1.0);
#endif
                depthAcc += output.depthAcc;
                depthWeight += output.depthWeight;
                primarySplit += output.numSamples;
                neeSplit += output.neeSamples;
                lpSplit += output.lpSamples;
                samplesTaken += 1;
            }
        }

#ifdef MARS_INCLUDE_AOVS
        m_config.statsImages->put(blocks);
#endif
        (*m_config.imageStatistics) += blockStatistics;
        m_config.imageStatistics->splatDepthAcc(depthAcc, depthWeight, primarySplit, samplesTaken, neeSplit, lpSplit);

        #if defined(MTS_DEBUG_FP)
            disableFPExceptions();
        #endif

        /* Make sure that there were no memory leaks */
        Assert(m_pool.unused());
    }

    Vector3 mapPointToUnitCube(const Scene *scene, const Point3 &point) const {
        AABB aabb = scene->getAABB();
        Vector3 size = aabb.getExtents();
        Vector3 result = point - aabb.min;
        for (int i = 0; i < 3; ++i)
            result[i] /= size[i];
        return result;
    }

    Point2 dirToCanonical(const Vector& d) const {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return {0, 0};
        }

        const Float cosTheta = std::min<Float>(std::max<Float>(d.z, -1.0f), 1.0f);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2.0 * M_PI;

        return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
    }

    int mapOutgoingDirectionToHistogramBin(const Vector3 &wo) const {
        const Point2 p = dirToCanonical(wo);
        const int res = Octtree::HISTOGRAM_RESOLUTION;
        const int result =
            std::min(int(p.x * res), res - 1) +
            std::min(int(p.y * res), res - 1) * res;
        return result;
    }

    /**
     * @brief
     *
     * @param sensorPath Currently traced path originating from the sensor
     * @param normWeight Throughput of current path, normalized by current pixelestimate.
     *                      Is modified by next intersection throughput if nextEdge is not nullptr
     * @param predVertex Previous vertex where the incoming ray originated from
     * @param predEdge Edge describing the incoming ray
     * @param curVertex Current vertex where the intersection we want to split is happening
     * @param trainingNodeLR Stores the looked-up training cache entry if not nullptr
     * @param trainingNodeNEE Stores the looked-up training cache entry if not nullptr
     * @param trainingNodeLP Stores the looked-up training cache entry if not nullptr
     * @param depth Depth of the current intersection (def: -1, meaning we use the sensorpaths depth)
     * @param isConnection Marks whether either edge is the new connection edge or not
     * @param nextEdge Direction the ray bounces in from the current vertex
     * @param nextVertex Intersection vertex of the bounced ray. We only need either edge or vertex, but take both here
     * @param stats If not nullptr, the splittingFactors are recorded in the stats
     *
     * @return SplitMapEntry containing all 3 different splittingFactors
     */
    SplitMapEntry
    getSFs(const Path &sensorPath, Spectrum& normWeight,
            const PathVertex *predVertex, const PathEdge *predEdge, PathVertex *curVertex,
            Octtree::TrainingNode **trainingNodeLR, Octtree::TrainingNode **trainingNodeNEE = nullptr,
            Octtree::TrainingNode **trainingNodeLP = nullptr, StatsRecursiveValues *stats=nullptr,
            const PathEdge *nextEdge=nullptr, const PathVertex *nextVertex=nullptr, int depth=-1, bool isConnection=false) const {

        if (depth == -1)         /* Record whether this was called routinely or to fill the remaining path */
            depth = sensorPath.edgeCount();
        if (depth < 2)           /* Supernode and Sample always have sF of 1 */
            return SplitMapEntry { 1.0 };

        /// Entry and aliases
        SplitMapEntry entry;
        Float &splittingFactorLR  = entry.LR;
        Float &splittingFactorNEE = entry.NEE;
        Float &splittingFactorLP  = entry.LP;

        /// Get some necessary constants
        const bool needsCaches = m_config.renderRRSMethod->needsTrainingPhase();
        const auto LR = Octtree::SplitType::LR;
        const auto NEE = Octtree::SplitType::NEE;
        const auto LP = Octtree::SplitType::LP;
        const Intersection &its = curVertex->getIntersection();
        const BSDF *bsdf = its.getBSDF();

        /// fetch some information about the BSDF
        const bool bsdfHasSmoothComponent = bsdf->getType() & BSDF::ESmooth;
        const Spectrum albedo = bsdf->getDiffuseReflectance(its) + bsdf->getSpecularReflectance(its);

        /* Look up the intersection point in our spatial cache */
        const int histogramBinIndex = mapOutgoingDirectionToHistogramBin(-predEdge->d);
        const auto mappedPoint = mapPointToUnitCube(m_scene.get(), its.p);

        /// Declare struct objects to fill
        const Octtree::SamplingNode *samplingNodeLR = nullptr;
        const Octtree::SamplingNode *samplingNodeNEE = nullptr;
        const Octtree::SamplingNode *samplingNodeLP = nullptr;
        Octtree::TrainingNode *l_trainingNodeLR  = nullptr;
        Octtree::TrainingNode *l_trainingNodeNEE = nullptr;
        Octtree::TrainingNode *l_trainingNodeLP  = nullptr;

        if (needsCaches) {
            m_config.cache->lookup(mappedPoint, histogramBinIndex, samplingNodeLR,  l_trainingNodeLR,  LR);
            m_config.cache->lookup(mappedPoint, histogramBinIndex, samplingNodeNEE, l_trainingNodeNEE, NEE);
            m_config.cache->lookup(mappedPoint, histogramBinIndex, samplingNodeLP,  l_trainingNodeLP,  LP);
            if (trainingNodeLR)
                *trainingNodeLR = l_trainingNodeLR;
            if (trainingNodeLP)
                *trainingNodeLP = l_trainingNodeLP;
            if (trainingNodeNEE)
                *trainingNodeNEE = l_trainingNodeNEE;
        }

#ifdef MARS_INCLUDE_AOVS
        /// update AOVs
        if (stats != nullptr && depth == 2) {
            stats->albedo.add(albedo);

            if (samplingNodeLR) {
                stats->lrFactorS.add(samplingNodeLR->earsFactorS[LR] * m_config.imageEarsFactor);
                stats->lrFactorR.add(samplingNodeLR->earsFactorR[LR] * m_config.imageEarsFactor);
                stats->lrEstimate.add(samplingNodeLR->estimate[LR]);
            }
            if (samplingNodeNEE) {
                stats->neeFactorS.add(samplingNodeNEE->earsFactorS[NEE] * m_config.imageEarsFactor);
                stats->neeFactorR.add(samplingNodeNEE->earsFactorR[NEE] * m_config.imageEarsFactor);
                stats->neeEstimate.add(samplingNodeNEE->estimate[NEE]);
            }
            if (samplingNodeLP) {
                stats->lpFactorS.add(samplingNodeLP->earsFactorS[LP] * m_config.imageEarsFactor);
                stats->lpFactorR.add(samplingNodeLP->earsFactorR[LP] * m_config.imageEarsFactor);
                stats->lpEstimate.add(samplingNodeLP->estimate[LP]);
            }
        }
#endif

        if (normWeight.isZero()) {
            /// Without any weight(contrib), tracing any further does not make any sense whatsoever
            entry = SplitMapEntry { 0.0 };
            return entry;

        } else {

            if (m_config.shareSF) {
                splittingFactorLR  =
                splittingFactorNEE =
                m_config.currentRRSMethod->evaluate(
                                                samplingNodeLR, m_config.imageEarsFactor,
                                                albedo, normWeight, bsdfHasSmoothComponent,
                                                depth, LR);
                splittingFactorLP = ! m_config.disableLP ?
                                        m_config.shareLP ? splittingFactorLR : 1.0
                                    : 0;
            } else {

                splittingFactorLR =
                    m_config.splitBSDF ?
                        m_config.currentRRSMethod->evaluate(
                            samplingNodeLR, m_config.imageEarsFactor,
                            albedo, normWeight, bsdfHasSmoothComponent,
                            depth, LR)
                        :
                        RRSMethod::BDPT().evaluate(
                            samplingNodeLR, m_config.imageEarsFactor,
                            albedo, normWeight, bsdfHasSmoothComponent,
                            depth, LR);

                splittingFactorNEE =
                    m_config.splitNEE ? m_config.currentRRSMethod->evaluate(
                            samplingNodeNEE, m_config.imageEarsFactor,
                            albedo, normWeight, bsdfHasSmoothComponent,
                            depth, NEE) : 1.0;

                splittingFactorLP =
                    ! m_config.disableLP ?
                        (m_config.splitLP ? m_config.currentRRSMethod->evaluate(
                            samplingNodeLP, m_config.imageEarsFactor,
                            albedo, normWeight, bsdfHasSmoothComponent,
                            depth, LP) : 1.0)
                    : 0;
            }
        }

        if (stats != nullptr) {
            stats->splittingFactorB.add(depth - 2, splittingFactorLR);
            stats->splittingFactorN.add(depth - 2, splittingFactorNEE);
            stats->splittingFactorL.add(depth - 2, splittingFactorLP);
        }

        if (nextEdge != nullptr) {
            const Float pdf = curVertex->evalPdf(m_scene.get(), predVertex, nextVertex, ERadiance, EMeasure::ESolidAngle);
            const Float invPDF = pdf > 1e-5f ? 1 / pdf : 0.0f;

            const Spectrum evald = isConnection ?
                curVertex->eval(m_scene.get(), predVertex, nextVertex, ERadiance, EMeasure::ESolidAngle, false) * invPDF :
                curVertex->weight[ERadiance];
#ifdef CHECK_SPLIT_FACTOR_CALC
            // For sensorpath, verify that the calculated splitting factor is the same as the cached one
            if (depth < (int) splitFactors.size()) {
                Assert(entry.isValid(m_config.currentRRSMethod->splittingMin, m_config.currentRRSMethod->splittingMax,
                             depth, m_config.currentRRSMethod->rrDepth, m_config.splitBSDF, m_config.splitLP, m_config.splitNEE, bsdfHasSmoothComponent));
                Assert(splitFactors[depth].epsEQ(entry));
                // Replace recalculated value by cached one to not propagate error
                entry = splitFactors[depth];
            }

            /* If the sampled component is not smooth we cannot recreate the bsdfWeight here.
            Otherwise, check that the recalculated value matches the cached one. */
            const uint sampledSmoothComponent = (curVertex->getComponentType() & (BSDF::ESmooth));
            if (sampledSmoothComponent) {
                const Spectrum recalcBSDFWeight = curVertex->eval(m_scene.get(), predVertex, nextVertex, ERadiance, EMeasure::ESolidAngle, false) * invPDF;
                // Assert (
                //     SPEC_EQ(recalcBSDFWeight, Spectrum(0.0f), 1e-2) ||  // Either we are extremely close to 0
                //     SPEC_EQ(recalcBSDFWeight, evald, 1e-2)              // Or we calculated the same bsdf weight as before
                // );
            }
#else
            Assert(depth+1 >= (int) splitFactors.size() && "Should never be called for sensorpath (except last node for connecting)");
#endif
            const auto bsdfWeight = evald * predEdge->weight[ERadiance];
            normWeight *= 1.f / entry.LR * bsdfWeight;
        }

        Assert(entry.isValid(m_config.currentRRSMethod->splittingMin, m_config.currentRRSMethod->splittingMax,
                             depth, m_config.currentRRSMethod->rrDepth, m_config.splitBSDF, m_config.splitLP, m_config.splitNEE, bsdfHasSmoothComponent));
        return entry;
    }

    std::tuple<Octtree::TrainingNode*, Octtree::TrainingNode*, Octtree::TrainingNode*>
    computeAndAddCurrentSFs(Path &sensorPath, const Spectrum& normWeight, StatsRecursiveValues &stats) const {
        const int depth = sensorPath.edgeCount();

        const PathVertex *predVertex = sensorPath.vertex(depth-1);
        const PathEdge *predEdge = sensorPath.edge(depth-1);
        PathVertex *curVertex = sensorPath.vertex(depth); // not const, need to change this iteration

        Octtree::TrainingNode *trainingNodeLR = nullptr;
        Octtree::TrainingNode *trainingNodeNEE = nullptr;
        Octtree::TrainingNode *trainingNodeLP = nullptr;

        /* normWeight is only changed by getSFs if nextEdge != nullptr. Since it is nullptr here, it won't be changed.
           Thus, it is safe to use const_cast here. */
        auto sfs = getSFs(sensorPath, const_cast<Spectrum&>(normWeight), predVertex, predEdge, curVertex,
                &trainingNodeLR, &trainingNodeNEE, &trainingNodeLP, &stats);

        splitFactors.emplace_back(sfs);
        Assert(trainingNodeLR  != nullptr || !m_config.splitBSDF
            && trainingNodeNEE != nullptr || !m_config.splitNEE
            && trainingNodeLP  != nullptr || !m_config.splitLP );
        return { trainingNodeLR, trainingNodeNEE, trainingNodeLP };
    }


    void performNEEandBDPTSamples(LiOutput &output, Octtree::TrainingNode* trainingNodeNEE, Octtree::TrainingNode* trainingNodeLP,
                        const Spectrum &normWeight, Path &emitterPath, Spectrum *importanceWeights,
                        Path &sensorPath, const Spectrum &absoluteWeight,
                        BMARSWorkResult *wr, StatsRecursiveValues &stats) const {

        const Float splittingFactorNEE = splitFactors.back().NEE;
        const Float splittingFactorLP = splitFactors.back().LP;

        if (splittingFactorNEE == 0 && splittingFactorLP == 0)
            return; // Nothing to be done here

        /// Get some necessary constants
        const Point2 &initialSamplePos = sensorPath.vertex(1)->getSamplePosition();
        const Float time = emitterPath.vertex(0)->getTime();
        const int depth = sensorPath.edgeCount();

        /* We will need an NEE emitterpath (path of length 1) for either case */
        const int NEE_EMITTER_DEPTH = 1;
        Path neeEmitterPath;
        neeEmitterPath.initialize(m_scene.get(), time, EImportance, m_pool);
        neeEmitterPath.randomWalk(m_scene.get(), m_sampler, NEE_EMITTER_DEPTH, m_config.currentRRSMethod->rrDepth,
                                ETransportMode::EImportance, m_pool);
        Assert(neeEmitterPath.length() == 1 && "Can emitter sample fail? Direct light sampling should only return 1 edge");
        /* NOTE: No cost since no ray has been traced */

        /* ==================================================================== */
        /*                              NEE SPLITS                              */
        /* ==================================================================== */
        {
            const auto NEE = Octtree::SplitType::NEE;
            const int EVAL_START = NOSELF;
            /// Sample accumulators
            Spectrum NEESum { 0.0 }; Spectrum NEESumSquares { 0.0 }; Float NEESumCosts { 0.f };
            Spectrum NEEEstimate { 0.0 }; Float NEECost { 0.0 };

            for (int sampleIndex = 0; sampleIndex < output.neeSamples; ++sampleIndex) {

                std::tie(NEEEstimate, NEECost) = bdptEvaluate(
                        sensorPath, absoluteWeight, normWeight, importanceWeights,
                        initialSamplePos, neeEmitterPath, wr, EVAL_START);

                output.emitted += NEEEstimate / splittingFactorNEE;
                stats.neeEmitted.add(depth - 1, absoluteWeight * NEEEstimate / splittingFactorNEE, 0);

#ifdef OUTLIER_MAX
                // clamp relative pixel contribution to OUTLIER_MAX (channel-wise average)
                Float outlierHeuristic = (NEEEstimate * normWeight).average();
                outlierHeuristic = std::max(Float(1), outlierHeuristic / OUTLIER_MAX);
                NEEEstimate /= outlierHeuristic;
#endif

                NEESum += NEEEstimate;
                NEESumSquares += NEEEstimate * NEEEstimate;
                NEESumCosts += NEECost;
            }
            // neeEmitterPath.release(m_pool); // No need to free, it is freed later anyways

            output.cost += NEESumCosts;

            if (trainingNodeNEE && output.neeSamples > 0) {
                trainingNodeNEE->splatEstimate(
                    NEESum,
                    NEESumSquares,
                    NEESumCosts,
                    output.neeSamples,
                    NEE
                );
            }
        }

        /* ==================================================================== */
        /*                           LIGHT PATH SPLITS                          */
        /* ==================================================================== */
        if (splittingFactorLP > 0) {
            const auto LP = Octtree::SplitType::LP;
            const int EVAL_START = NOSELF_NONEE;    // NEE is performed separately
            Path* localPath = nullptr;
            /// Sample accumulators
            Spectrum LPSum { 0.0 }; Spectrum LPSumSquares { 0.0 }; Float LPSumCosts { 0.0 };
            Spectrum LPEstimate { 0.0 }; Float LPCost { 0.0 };

            /// Finally split LP
            Assert(!m_config.disableLP || output.lpSamples == 0);
            for (int sampleIndex = 0; sampleIndex < output.lpSamples; ++sampleIndex) {

                /// Generate a new lightpath, IF REQUESTED!
                if (!m_config.currentRRSMethod->bmarsReuseLP) {
#ifdef BMARS_DEBUG_FP
                    disableFPExceptions();
#endif //BMARS_DEBUG_FP

                    neeEmitterPath.initialize(m_scene.get(), time, EImportance, m_pool); // initialize also frees
                    neeEmitterPath.randomWalk(m_scene.get(), m_sampler, m_emitterDepth,
                            m_config.currentRRSMethod->rrDepth, ETransportMode::EImportance, m_pool);
#ifdef BMARS_DEBUG_FP
                    enableFPExceptions();
#endif
                    LPSumCosts += (neeEmitterPath.edgeCount() - 1) * COST_BSDF;

                    importanceWeights[0] = Spectrum(1.0f);
                    for (size_t i = 1; i < neeEmitterPath.vertexCount(); ++i)
                        importanceWeights[i] = importanceWeights[i-1] *
                            neeEmitterPath.vertex(i-1)->weight[EImportance] *
                            neeEmitterPath.vertex(i-1)->rrWeight *
                            neeEmitterPath.edge(i-1)->weight[EImportance];

                    localPath = &neeEmitterPath;

                } else {
                    localPath = &emitterPath;
                }

                std::tie(LPEstimate, LPCost) = bdptEvaluate(
                        sensorPath, absoluteWeight, normWeight, importanceWeights,
                        initialSamplePos, *localPath, wr, EVAL_START);

                output.emitted += LPEstimate / splittingFactorLP;
                stats.lpEmitted.add(depth - 1, absoluteWeight * LPEstimate / splittingFactorLP, 0);

#ifdef OUTLIER_MAX
                // clamp relative pixel contribution to OUTLIER_MAX (channel-wise average)
                Float outlierHeuristic = (LPEstimate * normWeight).average();
                outlierHeuristic = std::max(Float(1), outlierHeuristic / OUTLIER_MAX);
                LPEstimate /= outlierHeuristic;
#endif

                LPSum += LPEstimate;
                LPSumSquares += LPEstimate * LPEstimate;
                LPSumCosts += LPCost;
            }

            output.cost += LPSumCosts;

            if (trainingNodeLP && output.lpSamples > 0) {
                trainingNodeLP->splatEstimate(
                    LPSum,
                    LPSumSquares,
                    LPSumCosts,
                    output.lpSamples,
                    LP
                );
            }
        }
        neeEmitterPath.release(m_pool);
    }


    void performLRSamples(LiOutput &output, Octtree::TrainingNode* trainingNode, const Spectrum &normWeight,
                        Path &emitterPath, Spectrum *importanceWeights,
                        Path &sensorPath, const Spectrum &absoluteWeight,
                        BMARSWorkResult *wr, StatsRecursiveValues &stats) const {
        /// Get some necessary constants
        const Point2 initialSamplePos = sensorPath.vertex(1)->getSamplePosition();
        const auto splittingFactorLR = splitFactors.back().LR;
        const int depth = sensorPath.edgeCount();
        const PathVertex *predVertex = sensorPath.vertex(depth-1);
        const PathEdge *predEdge = sensorPath.edge(depth-1);
        PathVertex *curVertex = sensorPath.vertex(depth); // not const, need to change this iteration

        /// Sample accumulators
        Spectrum lrSum { 0.0 }; Spectrum lrSumSquares { 0.0 }; Float lrSumCosts { 0.0 };

        for (int sampleIndex = 0; sampleIndex < output.numSamples; ++sampleIndex) {
            Spectrum LrEstimate { 0.0 }; Float LrCost { 0.0 };

            do {
                /* ==================================================================== */
                /*                             BSDF Sampling                            */
                /* ==================================================================== */
                PathVertex succVertex; PathEdge succEdge;

                Spectrum bsdfWeight { 1.0 };
                bool traced = false;   // Keep track of whether we've traced a ray for cost estimate
#ifdef BMARS_DEBUG_FP
                disableFPExceptions();
#endif // BMARS_DEBUG_FP
                if (!curVertex->sampleNext(m_scene.get(), m_sampler, predVertex, predEdge,
                            &succEdge, &succVertex, ERadiance, false, &bsdfWeight, &traced)) {
                    // Path terminated because sampling the next vertex and edge failed
                    // Sampling can fail even before a ray was traced, e.g., with a weight of 0
                    if (traced)
                        LrCost += COST_BSDF;
#ifdef BMARS_DEBUG_FP
                    enableFPExceptions();
#endif // BMARS_DEBUG_FP

                    break;  // Only break since we need to add to the output
                }
#ifdef BMARS_DEBUG_FP
                enableFPExceptions();
#endif // BMARS_DEBUG_FP
                /* When sampling was successful, a ray was certainly traced */
                Assert(traced);
                LrCost += COST_BSDF;

                /* Add vertex to sensorPath for MIS weights */
                sensorPath.append(&succEdge, &succVertex);

                /* Account for RRS in weight. BSDF weight already accounted for in sampleNext */
                Spectrum nestedAbsoluteWeight = absoluteWeight * bsdfWeight / splittingFactorLR;
                Spectrum nestedNormWeight = normWeight * bsdfWeight / splittingFactorLR;

                {
                    /// BSDF sampled light hits are evaluated here
                    const Intersection nestedIts = succVertex.getIntersection();
                    if (nestedIts.isEmitter()) {
                        Spectrum LrBSDFEstimate; Float LrBSDFCost;
                        // If we randomly hit an emitter we need to include it here for our LrEstimate
                        std::tie(LrBSDFEstimate, LrBSDFCost) = bdptEvaluate(
                                sensorPath, nestedAbsoluteWeight, nestedNormWeight, importanceWeights,
                                initialSamplePos, emitterPath, wr, SELF_ONLY);
                        Assert(LrBSDFCost == 0);
                        LrEstimate += bsdfWeight * LrBSDFEstimate;
                        stats.lrEmitted.add(depth - 1, nestedAbsoluteWeight * LrBSDFEstimate, 0);

                        const Emitter* emitter = nestedIts.shape->getEmitter();
                        if (emitter->isEnvironmentEmitter()) {
                            /// After an environment emitter, it makes no sense to continue tracing
                            sensorPath.dropLast();
                            break;
                        }
                    }
                }

                /* ==================================================================== */
                /*                         Indirect illumination                        */
                /* ==================================================================== */

                LiOutput outputNested = this->Li(sensorPath, emitterPath, nestedAbsoluteWeight, nestedNormWeight,
                                                    importanceWeights, wr, stats);
                LrEstimate += bsdfWeight * outputNested.totalContribution();
                LrCost += outputNested.cost;

                output.depthAcc += outputNested.depthAcc;
                output.depthWeight += outputNested.depthWeight;

                /* Reset path to before recursion state for next split sample */
                sensorPath.dropLast();
            } while (false);

            output.reflected += LrEstimate / splittingFactorLR;
            output.cost += LrCost;

#ifdef OUTLIER_MAX
            // clamp relative pixel contribution to OUTLIER_MAX (channel-wise average)
            Float outlierHeuristic = (LrEstimate * normWeight).average();
            outlierHeuristic = std::max(Float(1), outlierHeuristic / OUTLIER_MAX);
            LrEstimate /= outlierHeuristic;
#endif

            lrSum += LrEstimate;
            lrSumSquares += LrEstimate * LrEstimate;
            lrSumCosts += LrCost;
        }

        if (trainingNode && output.numSamples > 0) {
            trainingNode->splatEstimate(
                lrSum,
                lrSumSquares,
                lrSumCosts,
                output.numSamples,
                Octtree::SplitType::LR
            );
        }
    }

    LiOutput Li(Path &sensorPath, Path &emitterPath, const Spectrum& absoluteWeight, const Spectrum& normWeight,
                Spectrum *importanceWeights, BMARSWorkResult *wr, StatsRecursiveValues &stats) const {
        LiOutput output;

        const int depth = sensorPath.edgeCount();    // == vertexcount - 1
        Assert (depth >= 2);
        /* Since depth should be at least 2 when Li is called, nothing should be null */
        const PathVertex *curVertex = sensorPath.vertex(depth);

        if (m_config.maxDepth >= 0 && depth > m_config.maxDepth) {
            // maximum depth reached
            output.markAsLeaf(depth);
            return output;
        }

        /* ==================================================================== */
        /*                        Self-Emitted Radiance                         */
        /* ==================================================================== */

        if (depth == 2 && !m_config.hideEmitters) {
            /// Need to do this here for direct emitter hits
            /// For everything else, the splitting loop later will include the radiance
            const Intersection &its = curVertex->getIntersection();
            const Emitter* emitter = its.shape->getEmitter();
            Assert(its.shape && "If this was a nullptr, then sampleNext should have failed already");
            if (emitter) {
                Spectrum emitted; Float eCost;
                Assert(importanceWeights[0] == Spectrum(1.0f) && "Should ALWAYS be true");
                const Point2 initialSamplePos = sensorPath.vertex(1)->getSamplePosition();

                std::tie(emitted, eCost) = bdptEvaluate(
                        sensorPath, absoluteWeight, normWeight, importanceWeights,
                        initialSamplePos, emitterPath, wr, SELF_ONLY);

                Assert(eCost == 0 && "Self-Emittance shouldn't cost anything");
                output.emitted += emitted;
                stats.lrEmitted.add(depth-2, absoluteWeight * output.emitted, 0);
                if (emitter->isEnvironmentEmitter()) {
                    /// After an environment emitter, it makes no sense to continue tracing
                    output.markAsLeaf(depth);
                    return output;
                }
            }
        }

        /* ==================================================================== */
        /*                 Compute reflected radiance estimate                  */
        /* ==================================================================== */

        const auto trainingNodes = computeAndAddCurrentSFs(sensorPath, normWeight, stats);

        const auto splittingFactorLR = splitFactors.back().LR;
        const auto splittingFactorNEE = splitFactors.back().NEE;
        const auto splittingFactorLP = splitFactors.back().LP;

        int numSamplesBSDF, numSamplesNEE, numSamplesLP;
        if (m_config.shareSF) {
            // If sharing splittingFactors, we also want to round to the same number. Just like is done in EARS.
            numSamplesBSDF =
            numSamplesNEE = std::floor(splittingFactorLR + m_sampler->next1D());
            numSamplesLP = ! m_config.disableLP ?
                                        m_config.shareLP ? numSamplesBSDF : 1.0
                                    : 0;
        } else {
#ifdef LOW_DISCREPANCY_NUM_SAMPLES
            Float tmp = m_sampler->next1D();
            numSamplesBSDF = std::floor(tmp += splittingFactorLR);
            numSamplesNEE = std::floor(tmp += splittingFactorNEE - numSamplesBSDF);
            numSamplesLP = std::floor(tmp += splittingFactorLP - numSamplesNEE);
#else
            numSamplesBSDF = std::floor(splittingFactorLR + m_sampler->next1D());
            numSamplesNEE = std::floor(splittingFactorNEE + m_sampler->next1D());
            numSamplesLP = std::floor(splittingFactorLP + m_sampler->next1D());
#endif // LOW_DISCREPANCY_NUM_SAMPLES
        }

        output.numSamples = numSamplesBSDF;
        output.neeSamples = numSamplesNEE;
        output.lpSamples = numSamplesLP;

        /// compute LR splitting factor
        performLRSamples(output, std::get<0>(trainingNodes), normWeight,
                         emitterPath, importanceWeights, sensorPath,
                         absoluteWeight, wr, stats);
        /// compute NEE splitting factor and samples
        performNEEandBDPTSamples(output, std::get<1>(trainingNodes), std::get<2>(trainingNodes),
                            normWeight, emitterPath, importanceWeights, sensorPath,
                            absoluteWeight, wr, stats);

        /// Pop splitting factor that was added
        splitFactors.pop_back();

        if (output.depthAcc == 0) {
            /// all BSDF samples have failed :-(
            output.markAsLeaf(depth);
        }

        return output;
    }

    /**
     * @return int number of inserted splittingFactors
     */
    int computeAndCacheEmitterSplits(
            const Path &emitterPath, const PathEdge &connectionEdge, const Path &sensorPath,
            int s, Spectrum normThroughput) const {

        const int sizeBefore = splitFactors.size();
        int depth = (int) sensorPath.edgeCount();

#ifdef CHECK_SPLIT_FACTOR_CALC
        /* This block only verifies that the computation later (which is equivalent to this one, just
            for the emitterpath instead of the sensorpath) computes the correct result. We compute
            the splitting factors in the sensorpath again and compare whether we get the same result. */
        if (depth > 1) {
            /// We cannot really verify depth==1 since a new direct light sample is performed every call
            Spectrum fake_throughput = Spectrum(1.0f);
            if (!m_config.currentRRSMethod->useAbsoluteThroughput)
                fake_throughput /= m_pixelEstimate + Spectrum { 1e-2 };

            for (int i = 2; i < depth; i++) {
                const PathEdge *predEdge = sensorPath.edge(i - 1);
                const PathEdge *nextEdge = sensorPath.edge(i);
                const PathVertex *predVertex = sensorPath.vertex(i-1);
                PathVertex *curVertex = sensorPath.vertex(i);
                const PathVertex *nextVertex = sensorPath.vertex(i+1);

                getSFs(sensorPath, fake_throughput, predVertex, predEdge, curVertex,
                        nullptr, nullptr, nullptr, nullptr, // Not called during normal sampling routine -> nullptrs
                        nextEdge, nextVertex, i);
            }
            Assert(SPEC_EQ(fake_throughput, normThroughput));
        }
#endif // CHECK_SPLIT_FACTOR_CALC

        if (connectionEdge.length == 0) {
            // In this case vs/vt is a supernode
            Assert(sensorPath.vertex(depth)->isSensorSupernode() || emitterPath.vertex(s)->isEmitterSupernode());
            Assert(s==0 || depth==0);
            goto EmitterPathEnd;
        }

        if (s > 1) {
            /* Multiply the throughput by the inverse splitting factor and the bsdfWeight (like in Li) */
            const PathVertex *preVertex = sensorPath.vertexOrNull(depth-1);
            PathVertex *curVertex = sensorPath.vertex(depth);
            const PathVertex *nextVertex = emitterPath.vertex(s);
            const PathEdge *predEdge = sensorPath.edge(depth-1);
            const PathEdge *nextEdge = &connectionEdge;

            /* We need this recalculation since it changes the throughput */
            // Assert(depth < (int) splitFactors.size()); // Make sure we land in the check whether the recomputed matches cache
            Assert(depth + 1 == (int) splitFactors.size()); // Number of splitfactors should match
            getSFs(sensorPath, normThroughput, preVertex, predEdge, curVertex,
                    nullptr, nullptr, nullptr, nullptr, // Not called during normal sampling routine -> nullptrs
                    nextEdge, nextVertex, depth, true);

            /// Now compute and cache all emitter split factors
            for (int i = s; i > 1; --i) {
                ++depth;
                predEdge  = nextEdge;
                preVertex = curVertex;

                const bool isConnection = (i == s);
                curVertex = emitterPath.vertex(i);
                nextVertex = emitterPath.vertex(i-1);
                nextEdge = emitterPath.edge(i-1);

                auto sfs = getSFs(sensorPath, normThroughput, preVertex, predEdge, curVertex,
                                nullptr, nullptr, nullptr, nullptr, // Not called during normal sampling routine -> nullptrs
                                nextEdge, nextVertex, depth, isConnection);

                splitFactors.emplace_back(sfs);
            }
        }

EmitterPathEnd:
        if (s >= 1)
            splitFactors.emplace_back(1.0); // emitterPath.vertex(1) -> EmitterSample
        splitFactors.emplace_back(1.0);     // emitterPath.vertex(0) -> EmitterSupernode
        return splitFactors.size() - sizeBefore;
    }

    /* Adjusted Path::miWeight function to include splittingFactor */
    Float miWeightSplitting(
            const Path &emitterSubpath, const PathEdge *connectionEdge, const Path &sensorSubpath,
            int s, const int t) const {
        const Scene *scene = m_scene.get();
        const bool &lightImage = m_config.lightImage;

        bool sampleDirect = m_config.sampleDirect;
        int k = s+t+1, n = k+1;

        const PathVertex
                *vsPred = emitterSubpath.vertexOrNull(s-1),
                *vtPred = sensorSubpath.vertexOrNull(t-1),
                *vs = emitterSubpath.vertex(s),
                *vt = sensorSubpath.vertex(t);

        /* pdfImp[i] and pdfRad[i] store the area/volume density of vertex
        'i' when sampled from the adjacent vertex in the emitter
        and sensor direction, respectively. */

        Float ratioEmitterDirect = 0.0f, ratioSensorDirect = 0.0f;
        Float *pdfImp      = (Float *) alloca(n * sizeof(Float)),
              *pdfRad      = (Float *) alloca(n * sizeof(Float));
        bool  *connectable = (bool *)  alloca(n * sizeof(bool)),
              *isNull      = (bool *)  alloca(n * sizeof(bool));

        /* Keep track of which vertices are connectable / null interactions */
        int pos = 0;
        for (int i=0; i<=s; ++i) {
            const PathVertex *v = emitterSubpath.vertex(i);
            connectable[pos] = v->isConnectable();
            isNull[pos] = v->isNullInteraction() && !connectable[pos];
            pos++;
        }

        for (int i=t; i>=0; --i) {
            const PathVertex *v = sensorSubpath.vertex(i);
            connectable[pos] = v->isConnectable();
            isNull[pos] = v->isNullInteraction() && !connectable[pos];
            pos++;
        }

        if (k <= 3)
            sampleDirect = false;

        EMeasure vsMeasure = EArea, vtMeasure = EArea;
        if (sampleDirect) {
            /* When direct sampling is enabled, we may be able to create certain
            connections that otherwise would have failed (e.g. to an
            orthographic camera or a directional light source) */
            const AbstractEmitter *emitter = (s > 0 ? emitterSubpath.vertex(1) : vt)->getAbstractEmitter();
            const AbstractEmitter *sensor = (t > 0 ? sensorSubpath.vertex(1) : vs)->getAbstractEmitter();

            EMeasure emitterDirectMeasure = emitter->getDirectMeasure();
            EMeasure sensorDirectMeasure  = sensor->getDirectMeasure();

            connectable[0]   = emitterDirectMeasure != EDiscrete && emitterDirectMeasure != EInvalidMeasure;
            connectable[1]   = emitterDirectMeasure != EInvalidMeasure;
            connectable[k-1] = sensorDirectMeasure != EInvalidMeasure;
            connectable[k]   = sensorDirectMeasure != EDiscrete && sensorDirectMeasure != EInvalidMeasure;

            /* The following is needed to handle orthographic cameras &
            directional light sources together with direct sampling */
            if (t == 1)
                vtMeasure = sensor->needsDirectionSample() ? EArea : EDiscrete;
            else if (s == 1)
                vsMeasure = emitter->needsDirectionSample() ? EArea : EDiscrete;
        }

        /* Collect importance transfer area/volume densities from vertices */
        pos = 0;
        pdfImp[pos++] = 1.0;        // pos=0

        /* If 'pos > emitterDepth', the path couldn't have produced by ligh tracing => PDF = 0 */
        for (int i=0; i<s; ++i)
            pdfImp[pos++] = emitterSubpath.vertex(i)->pdf[EImportance]      // pos in [1, s]
                * emitterSubpath.edge(i)->pdf[EImportance];

        pdfImp[pos] = (pos > m_emitterDepth) ? 0.0 :    // pos = s+1
            vs->evalPdf(scene, vsPred, vt, EImportance, vsMeasure) * connectionEdge->pdf[EImportance];
        pos++;

        if (t > 0) {
            pdfImp[pos] = (pos > m_emitterDepth) ? 0.0 :    // pos = s+2
                vt->evalPdf(scene, vs, vtPred, EImportance, vtMeasure) * sensorSubpath.edge(t-1)->pdf[EImportance];
            pos++;

            for (int i=t-1; i>0; --i) {
                pdfImp[pos] = (pos > m_emitterDepth) ? 0.0 : // pos in [s+3; s+t+1]
                    sensorSubpath.vertex(i)->pdf[EImportance] * sensorSubpath.edge(i-1)->pdf[EImportance];
                pos++;
            }
        }

        /* Collect radiance transfer area/volume densities from vertices */
        pos = 0;
        if (s > 0) {
            for (int i=0; i<s-1; ++i)
                pdfRad[pos++] = emitterSubpath.vertex(i+1)->pdf[ERadiance]
                    * emitterSubpath.edge(i)->pdf[ERadiance];

            pdfRad[pos++] = vs->evalPdf(scene, vt, vsPred, ERadiance, vsMeasure)
                * emitterSubpath.edge(s-1)->pdf[ERadiance];
        }

        pdfRad[pos++] = vt->evalPdf(scene, vtPred, vs, ERadiance, vtMeasure)
            * connectionEdge->pdf[ERadiance];

        for (int i=t; i>0; --i)
            pdfRad[pos++] = sensorSubpath.vertex(i-1)->pdf[ERadiance]
                * sensorSubpath.edge(i-1)->pdf[ERadiance];

        pdfRad[pos++] = 1.0;

        /* Multiply sF onto forward pdf's to account for them in MIS
           pdfRad is ordered light->cam, splitFactors cam->light.
           Also, pdfRad is shifted by 2. Thus: '(n-2)-i'*/
        if (m_config.budgetAware && m_config.currentRRSMethod->technique == RRSMethod::EBMARS) {
            Assert((int) splitFactors.size() == n);
            for (int i = 0; i <= n-2; i++) {
                pdfRad[i] *= pow(splitFactors[(n-2)-i].LR, m_config.sfExp / 2);
            }
        }

        /* When the path contains specular surface interactions, it is possible
        to compute the correct MI weights even without going through all the
        trouble of computing the proper generalized geometric terms (described
        in the SIGGRAPH 2012 specular manifolds paper). The reason is that these
        all cancel out. But to make sure that that's actually true, we need to
        convert some of the area densities in the 'pdfRad' and 'pdfImp' arrays
        into the projected solid angle measure */
        for (int i=1; i <= k-3; ++i) {
            if (i == s || !(connectable[i] && !connectable[i+1]))
                continue;

            const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k-i);
            const PathVertex *succ = i+1 <= s ? emitterSubpath.vertex(i+1) : sensorSubpath.vertex(k-i-1);
            const PathEdge *edge = i < s ? emitterSubpath.edge(i) : sensorSubpath.edge(k-i-1);

            pdfImp[i+1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface()  ? dot(edge->d, cur->getGeometricNormal())  : 1));
        }

        for (int i=k-1; i >= 3; --i) {
            if (i-1 == s || !(connectable[i] && !connectable[i-1]))
                continue;

            const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k-i);
            const PathVertex *succ = i-1 <= s ? emitterSubpath.vertex(i-1) : sensorSubpath.vertex(k-i+1);
            const PathEdge *edge = i <= s ? emitterSubpath.edge(i-1) : sensorSubpath.edge(k-i);

            pdfRad[i-1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface()  ? dot(edge->d, cur->getGeometricNormal())  : 1));
        }

        int emitterRefIndirection = 2, sensorRefIndirection = k-2;

        /* One more array sweep before the actual useful work starts -- phew! :)
        "Collapse" edges/vertices that were caused by BSDF::ENull interactions.
        The BDPT implementation is smart enough to connect straight through those,
        so they shouldn't be treated as Dirac delta events in what follows */
        for (int i=1; i <= k-3; ++i) {
            if (!connectable[i] || !isNull[i+1])
                continue;

            int start = i+1, end = start;
            while (isNull[end+1])
                ++end;

            if (!connectable[end+1]) {
                /// The chain contains a non-ENull interaction
                isNull[start] = false;
                continue;
            }

            const PathVertex *before = i     <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k-i);
            const PathVertex *after  = end+1 <= s ? emitterSubpath.vertex(end+1) : sensorSubpath.vertex(k-end-1);

            Vector d = before->getPosition() - after->getPosition();
            Float lengthSquared = d.lengthSquared();
            d /= std::sqrt(lengthSquared);

            Float geoTerm = std::abs(
                (before->isOnSurface() ? dot(before->getGeometricNormal(), d) : 1) *
                (after->isOnSurface()  ? dot(after->getGeometricNormal(),  d) : 1)) / lengthSquared;

            pdfRad[start-1] *= pdfRad[end] * geoTerm;
            pdfRad[end] = 1;
            pdfImp[start] *= pdfImp[end+1] * geoTerm;
            pdfImp[end+1] = 1;

            /* When an ENull chain starts right after the emitter / before the sensor,
            we must keep track of the reference vertex for direct sampling strategies. */
            if (start == 2)
                emitterRefIndirection = end + 1;
            else if (end == k-2)
                sensorRefIndirection = start - 1;

            i = end;
        }

        double initial = 1.0f;

        /* When direct sampling strategies are enabled, we must
        account for them here as well */
        if (sampleDirect) {
            /* Direct connection probability of the emitter */
            const PathVertex *sample = s>0 ? emitterSubpath.vertex(1) : vt;
            const PathVertex *ref = emitterRefIndirection <= s
                ? emitterSubpath.vertex(emitterRefIndirection) : sensorSubpath.vertex(k-emitterRefIndirection);
            EMeasure measure = sample->getAbstractEmitter()->getDirectMeasure();

            if (connectable[1] && connectable[emitterRefIndirection])
                ratioEmitterDirect = ref->evalPdfDirect(scene, sample, EImportance,
                    measure == ESolidAngle ? EArea : measure) / pdfImp[1];

            /* Direct connection probability of the sensor */
            sample = t>0 ? sensorSubpath.vertex(1) : vs;
            ref = sensorRefIndirection <= s ? emitterSubpath.vertex(sensorRefIndirection)
                : sensorSubpath.vertex(k-sensorRefIndirection);
            measure = sample->getAbstractEmitter()->getDirectMeasure();

            if (connectable[k-1] && connectable[sensorRefIndirection])
                ratioSensorDirect = ref->evalPdfDirect(scene, sample, ERadiance,
                    measure == ESolidAngle ? EArea : measure) / pdfRad[k-1];

            if (s == 1)
                initial = ratioEmitterDirect != 0 ? (initial / ratioEmitterDirect) : std::numeric_limits<Float>::infinity();
            else if (t == 1)
                initial /= ratioSensorDirect;
        }

        /// Compute MIS-sf weight of the methods combination that generated this path
        const bool includeSF = m_config.budgetAware && m_config.currentRRSMethod->technique == RRSMethod::EBMARS;
        const double currMethodSF = [&](){
            if (includeSF) {
                Assert(m_config.splitLP || splitFactors[n-(s+2)].LP == 1 || splitFactors[n-(s+2)].LP == 0);
                Assert(m_config.splitNEE || splitFactors[n-3].NEE == 1 || splitFactors[n-3].NEE == 0);
                Assert(splitFactors[n-1].NEE == 1 && splitFactors[n-2].NEE == 1);
                const auto sf = double((s == 1) ?
                            splitFactors[n-3].NEE :  // Splitting NEE, if disabled will be 0/1
                            splitFactors[n-(s+2)].LP // Splitting LP,  if disabled will be 0/1
                        );
                return pow(sf, m_config.sfExp);
            } else {
                return 1.0; // Not splitting NEE/LP -> Always factor 1.0
            }
        }();


        double weight = currMethodSF, pdf = initial;

        /* With all of the above information, the MI weight can now be computed.
        Since the goal is to evaluate the power heuristic, the absolute area
        product density of each strategy is interestingly not required. Instead,
        an incremental scheme can be used that only finds the densities relative
        to the (s,t) strategy, which can be done using a linear sweep. For
        details, refer to the Veach thesis, p.306. */

        for (int i=s+1; i<k; ++i) {
            const double local_sF = [&](){
                /* We need to check this since even when sNEE/sLP are disabled,
                   these might be 0, which should not be included in the MIS weight. */
                if (includeSF) {
                    Assert(m_config.splitLP  || splitFactors[n-(i+2)].LP == 1 || splitFactors[n-(i+2)].LP == 0);
                    const auto sf = double((i == 1) ?
                                splitFactors[n-3].NEE :
                                splitFactors[n-(i+2)].LP
                            );
                    return pow(sf, m_config.sfExp);
                } else {
                    return 1.0;
                }
            }();

            double next = pdfImp[i] == 0 ? 0 : pdf * (double) pdfImp[i] / (double) pdfRad[i],
                value = next;

            if (sampleDirect) {
                if (i == 1)
                    value *= ratioEmitterDirect;
                else if (i == sensorRefIndirection)
                    value *= ratioSensorDirect;
            }

            int tPrime = k-i-1;
            if (connectable[i] && (connectable[i+1] || isNull[i+1]) && (lightImage || tPrime > 1)) {
                weight += value*value * local_sF;
            }

            pdf = next;
        }

        /* As above, but now compute pdf[i] with i<s (this is done by
        evaluating the inverse of the previous expressions). */
        pdf = initial;
        for (int i=s-1; i>=0; --i) {
            const double local_sF = [&](){
                /// Same as above.
                if (includeSF) {
                    Assert(m_config.splitLP  || splitFactors[n-(i+2)].LP == 1 || splitFactors[n-(i+2)].LP == 0);
                    const auto sf = double((i == 1) ?
                                splitFactors[n-3].NEE :
                                splitFactors[n-(i+2)].LP
                            );
                    return pow(sf, m_config.sfExp);
                } else {
                    return 1.0;
                }
            }();

            double next = pdfRad[i+1] == 0 ? 0 : pdf * (double) pdfRad[i+1] / (double) pdfImp[i+1],
                value = next;

            if (sampleDirect) {
                if (i == 1)
                    value *= ratioEmitterDirect;
                else if (i == sensorRefIndirection)
                    value *= ratioSensorDirect;
            }

            int tPrime = k-i-1;
            if (connectable[i] && (connectable[i+1] || isNull[i+1]) && (lightImage || tPrime > 1)) {
                weight += value*value * local_sF;
            }

            pdf = next;
        }

        /***
         *  Weight can be 0 if we are including the splitting factors.
         *  Simple reason is: MARS disables NEE/LP tracing when the BSDF has no smooth component
         *  Whenever weight is 0, this means none of the possible paths would have any contribution
         *  Therefore, setting the returned MIS weight to 0 should be fine here.
         */
        Assert(weight != 0 || !m_config.splittingFactorMIS);
        return (Float) ( weight == 0 ? 0.0 : currMethodSF / weight );
    }

    /***
     *  bdpt evaluate function just adapted to only iterate over emitterpath
     *
     *  minS:
     *      0 for self-emitted light only,
     *      1 to connect whole light path including NEE
     *      2 to connect whole light path EXCLUDING NEE
     */
    std::pair<Spectrum, Float> bdptEvaluate(
            Path &sensorPath, const Spectrum &absoluteWeight, const Spectrum &normWeight,
            const Spectrum *importanceWeights, const Point2 initialSamplePos, Path &emitterPath,
            BMARSWorkResult *wr, const int minS) const {
        const Scene *scene = m_scene.get();
        const bool selfEmittedOnly = (minS == 0);
        Spectrum LEstimateBDPT(0.0f);
        PathVertex tempEndpoint, tempSample;
        PathEdge tempEdge, connectionEdge;

        const int t = sensorPath.edgeCount(); // == vertexcount - 1

        /* Since this function is called for lightImage, the edges and vertices can actually be nullptr */
        PathVertex *predVertex = sensorPath.vertexOrNull(t-1);
        PathVertex *curVertex  = sensorPath.vertexOrNull(t);
        PathEdge *predEdge   = sensorPath.edgeOrNull(t-1);

        Assert(m_config.lightImage || (predVertex != nullptr && curVertex != nullptr && predEdge != nullptr));

        /// Make sure our maximum depth is accounted for. Needed for MIS weights
        int maxS = (int) emitterPath.vertexCount() - 1;
        if (m_config.maxDepth >= 0)
            maxS = std::min(maxS, m_config.maxDepth + 1 - t);

        /// If only emitted light should be estimated, run the following loop only with the EmitterSupernode
        if (selfEmittedOnly) {
            maxS = 0;
            /* HACK: For self-emittance of the BSDF sampled surface, the splittingFactor
            has to be set to 1.0 since s=0, computeAndCacheSF will only append 1.0 once
            (for the supernode), but we also need it for the current 'EmitterSample' */
            splitFactors.emplace_back(1.0);
        }

        /// For every vertex in the sensorpath, there has to be a splittingFactor inserted already
        const int splitFsLenBefore = splitFactors.size();
        Assert(splitFsLenBefore == t + 1);

        Spectrum norm_radianceWeight { normWeight };

        PathVertex *castTemp = nullptr;
        for (int s = maxS; s >= minS; --s) {
            PathVertex
                *vsPred = emitterPath.vertexOrNull(s-1),
                *vtPred = predVertex,
                *vs = emitterPath.vertex(s),
                *vt = curVertex;
            PathEdge
                *vsEdge = emitterPath.edgeOrNull(s-1),
                *vtEdge = predEdge;

            RestoreMeasureHelper rmh0(vs), rmh1(vt);

            /* Will be set to true if direct sampling was used */
            bool sampleDirect = false;

            /* Stores the pixel position associated with this sample */
            Point2 samplePos = initialSamplePos;

            /* Allowed remaining number of ENull vertices that can
                   be bridged via pathConnect (negative=arbitrarily many) */
            int remaining = m_config.maxDepth - s - t + 1;

            /* Will receive the path weight of the (s, t)-connection */
            Spectrum value;

            if (vs->isEmitterSupernode()) {
                Assert(selfEmittedOnly);
                castTemp = vt->clone(m_pool);
                if (!vt->cast(scene, PathVertex::EEmitterSample) || vt->isDegenerate())
                    continue;

                value =
                    vs->eval(scene, vsPred, vt, EImportance) *
                    vt->eval(scene, vtPred, vs, ERadiance);
            } else if (vt->isSensorSupernode()) {
                /* If possible, convert 'vs' into an sensor sample */
                if (!vs->cast(scene, PathVertex::ESensorSample) || vs->isDegenerate())
                    continue;
                /* Make note of the changed pixel sample position */
                if (!vs->getSamplePosition(vsPred, samplePos))
                    continue;

                value = importanceWeights[s] *
                    vs->eval(scene, vsPred, vt, EImportance) *
                    vt->eval(scene, vtPred, vs, ERadiance);
            } else if (m_config.sampleDirect && ((t == 1 && s > 1) || (s == 1 && t > 1))) {
                /* s==1/t==1 path: use a direct sampling strategy if requested */
                if (s == 1) {
                    if (vt->isDegenerate())
                        continue;
                    /* Generate a position on an emitter using direct sampling */
#ifdef BMARS_DEBUG_FP
                    disableFPExceptions();
#endif // BMARS_DEBUG_FP
                    value = vt->sampleDirect(scene, m_sampler.get(), &tempEndpoint, &tempEdge, &tempSample, EImportance);
#ifdef BMARS_DEBUG_FP
                    enableFPExceptions();
#endif // BMARS_DEBUG_FP
                    if (value.isZero())
                        continue;
                    vs = &tempSample; vsPred = &tempEndpoint; vsEdge = &tempEdge;
                    value *= vt->eval(scene, vtPred, vs, ERadiance);
                    vt->measure = EArea;
                } else {
                    if (vs->isDegenerate())
                        continue;
                    /* Generate a position on the sensor using direct sampling */
                    value = importanceWeights[s] * vs->sampleDirect(scene, m_sampler.get(),
                        &tempEndpoint, &tempEdge, &tempSample, ERadiance);
                    if (value.isZero())
                        continue;
                    vt = &tempSample; vtPred = &tempEndpoint; vtEdge = &tempEdge;
                    value *= vs->eval(scene, vsPred, vt, EImportance);
                    vs->measure = EArea;
                }

                sampleDirect = true;
            } else {
                /* Can't connect degenerate endpoints */
                if (vs->isDegenerate() || vt->isDegenerate())
                    continue;

                value = importanceWeights[s] *
                    vs->eval(scene, vsPred, vt, EImportance) *
                    vt->eval(scene, vtPred, vs, ERadiance);

                /* Temporarily force vertex measure to EArea. Needed to
                    handle BSDFs with diffuse + specular components */
                vs->measure = vt->measure = EArea;
            }

            /* Attempt to connect the two endpoints, which could result in
                the creation of additional vertices (index-matched boundaries etc.) */
            int interactions = remaining; // backup
#ifdef BMARS_DEBUG_FP
            disableFPExceptions();
#endif // BMARS_DEBUG_FP

            if (value.isZero() || !connectionEdge.pathConnectAndCollapse(
                    scene, vsEdge, vs, vt, vtEdge, interactions)) {
#ifdef BMARS_DEBUG_FP
                enableFPExceptions();
#endif // BMARS_DEBUG_FP
                continue;
            }
#ifdef BMARS_DEBUG_FP
            enableFPExceptions();
#endif // BMARS_DEBUG_FP

            /* Account for the terms of the measurement contribution
                function that are coupled to the connection edge */
            if (!sampleDirect)
                value *= connectionEdge.evalCached(vs, vt, PathEdge::EGeneralizedGeometricTerm);
            else
                value *= connectionEdge.evalCached(vs, vt, PathEdge::ETransmittance |
                        (s == 1 ? PathEdge::ECosineRad : PathEdge::ECosineImp));

            if (sampleDirect) {
                /* A direct sampling strategy was used, which generated
                    two new vertices at one of the path ends. Temporarily
                    modify the path to reflect this change */
                if (t == 1) {
                    /* Determine the pixel sample position, we need it for our pixelEstimate */
                    if (vt->isSensorSample() && !vt->getSamplePosition(vs, samplePos))
                        continue;

                    /* No need to include splitting factor in computation as it is 1 for t = 0/1
                       Probably also don't need the weight recalc as it's always 1 for us, but just to be safe */
                    norm_radianceWeight = vtPred->weight[ERadiance] * vtEdge->weight[ERadiance];
                    sensorPath.swapEndpoints(vtPred, vtEdge, vt);

                    if (!m_config.currentRRSMethod->useAbsoluteThroughput) {
                        Spectrum local_pixelEstimate { 1e-2f };
                        Assert(wr->getLightImage()->getOffset() == Point2i(0) && wr->getLightImage()->getBorderSize() == 1);
                        if (m_config.pixelEstimate->get()) {
                            // Get clamped position for pixelestimate at fractional position samplePos
                            Point2i estimatePos { samplePos };
                            estimatePos.x = std::max(std::min((*m_config.pixelEstimate)->getWidth() - 1, estimatePos.x), 0);
                            estimatePos.y = std::max(std::min((*m_config.pixelEstimate)->getHeight() - 1, estimatePos.y), 0);
                            local_pixelEstimate += (*m_config.pixelEstimate)->getPixel(estimatePos);
                        }
                        norm_radianceWeight /= local_pixelEstimate;
                    }
                } else {
                    emitterPath.swapEndpoints(vsPred, vsEdge, vs);
                }
            }

#ifndef CHECK_SPLIT_FACTOR_CALC
            if (m_config.budgetAware) {
#endif
                /* Fill splitting factor map with correct splitting factors for light path connection */
                int numInserted = computeAndCacheEmitterSplits(
                        emitterPath, connectionEdge, sensorPath,
                        s, norm_radianceWeight
                    );
                Assert(numInserted == s+1);
#ifndef CHECK_SPLIT_FACTOR_CALC
            }
#endif

            /* Compute the multiple importance sampling weight */
            Float miWeight = miWeightSplitting(emitterPath, &connectionEdge,
                sensorPath, s, t);

#ifndef CHECK_SPLIT_FACTOR_CALC
            if (m_config.budgetAware) {
#endif
                for (int j = 0; j < s+1; j++)
                    splitFactors.pop_back();
                Assert(splitFsLenBefore == (int) splitFactors.size());
#ifndef CHECK_SPLIT_FACTOR_CALC
            }
#endif

            if (sampleDirect) {
                /* Now undo the previous change */
                if (t == 1) {
                    norm_radianceWeight = normWeight;
                    sensorPath.swapEndpoints(vtPred, vtEdge, vt);
                } else
                    emitterPath.swapEndpoints(vsPred, vsEdge, vs);
            }

            /* Determine the pixel sample position when necessary */
            if (vt->isSensorSample() && !vt->getSamplePosition(vs, samplePos))
                continue;

            #if BDPT_DEBUG == 1
                /* When the debug mode is on, collect samples
                    separately for each sampling strategy. Note: the
                    following piece of code artificially increases the
                    exposure of longer paths */
                Spectrum splatValue = value * (m_config.showWeighted
                    ? miWeight : 1.0f);// * std::pow(2.0f, s+t-3.0f));
                wr->putDebugSample(s, t, samplePos, splatValue);
            #endif

            if (t >= 2) {
                LEstimateBDPT += value * miWeight;
            } else {
                Assert(vt->getType() != PathVertex::EEmitterSample);
                wr->putLightSample(samplePos, absoluteWeight * value * miWeight);
            }
        }

        if (castTemp != nullptr) {
            *curVertex = *castTemp;
            m_pool.release(castTemp);
        }

        /// Revert emitter hack
        if (selfEmittedOnly)
            splitFactors.pop_back();

        // maxS <= vertexCount-1 => emitted light does not add cost since emittance is done via EmitterSupernode Vertex
        return { LEstimateBDPT, maxS * COST_NEE };
    }

    std::pair<Point2, LiOutput> EARSWalkAndEvaluate(
        Path &emitterPath,
        const Point2i &offset,
        const Float time,
        BMARSWorkResult *result,
        OutlierRejectedAverage &blockStatistics,
        StatsRecursiveValues &stats
    ) const {
        const Spectrum metricNorm = m_config.renderRRSMethod->useAbsoluteThroughput ? Spectrum { 1.0 } :
                                                           (m_pixelEstimate + Spectrum { 1e-2 });
        const Spectrum expectedContribution = m_pixelEstimate / metricNorm;

        /* Precompute importance weights necessary for BDPT connections */
        Assert(m_emitterDepth+1 >= (int) emitterPath.vertexCount());
        const size_t allocSize = m_config.currentRRSMethod->bmarsReuseLP ?
                                 emitterPath.vertexCount() : (m_emitterDepth+1);
        Spectrum *importanceWeights = (Spectrum *) alloca(allocSize * sizeof(Spectrum));
        importanceWeights[0] = Spectrum { 1.0 };
        for (size_t i = 1; i < emitterPath.vertexCount(); ++i)
            importanceWeights[i] = importanceWeights[i-1] *
                emitterPath.vertex(i-1)->weight[EImportance] *
                emitterPath.vertex(i-1)->rrWeight *
                emitterPath.edge(i-1)->weight[EImportance];

        /* sensorPath sampling similar to randomWalkFromPixel */
        Path sensorPath;
        sensorPath.initialize(m_scene.get(), time, ERadiance, m_pool);
        LiOutput output;
        PathVertex v1, v2, *v0 = sensorPath.vertex(0);
        PathEdge e0, e1;
        Point2 initialSamplePos{ -1.0 };

        const int t = v0->sampleSensor(m_scene.get(), m_sampler, offset, &e0, &v1, &e1, &v2);
        splitFactors.emplace_back(1.0);     // v0 - Supernode
        Spectrum absoluteWeight { 0.0 };
        /* Append the successfully sampled sensor vertices and edges */
        if (t >= 1) {
            initialSamplePos = v1.getSamplePosition();
            sensorPath.append(&e0, &v1);
            splitFactors.emplace_back(1.0); // v1 - camera plane
        }
        if (t == 2) {
            sensorPath.append(&e1, &v2);
            /* v2's splittingFactor is saved in the Li call */
        }
#ifdef BMARS_DEBUG_FP
        enableFPExceptions();
#endif //BMARS_DEBUG_FP
        /* If a scene intersection was found during sensor sampling continue with an EARS walk */
        if (t == 2) {
            /* If this was not 1.0, the following calculations would be missing the rr factor */
            Assert(v0->rrWeight == Float(1.0f) && sensorPath.vertex(1)->rrWeight == Float(1.0f));

            /* Prepare weight and start recursing into the EARS walk */
            absoluteWeight = v0->weight[ERadiance] * e0.weight[ERadiance] *
                             v1.weight[ERadiance] * e1.weight[ERadiance];
            Spectrum normWeight { absoluteWeight };
            if (!m_config.currentRRSMethod->useAbsoluteThroughput)
                normWeight /= m_pixelEstimate + Spectrum { 1e-2 };

            output = Li(sensorPath, emitterPath, absoluteWeight, normWeight,
                        importanceWeights, result, stats);

            Assert(sensorPath.vertexCount() == 3 && sensorPath.edgeCount() == 2 &&
                "sensorPath should only contain v0, v1, v2, e0, e1 here. Forgot a dropLast anywhere?");
            Assert(splitFactors.size() == 2);

            /* Drop e1 and v2 */
            sensorPath.dropLast();
        }

        /* Perform lightImage calculation if requested */
        if (m_config.lightImage) {
            Spectrum LrEstimateBDPT;
            Float LrCostBDPT;

            if (!m_config.currentRRSMethod->bmarsReuseLP) {
                const Float time = emitterPath.vertex(0)->getTime();
                emitterPath.release(m_pool);
                emitterPath.initialize(m_scene.get(), time, EImportance, m_pool);
                emitterPath.randomWalk(m_scene.get(), m_sampler, m_emitterDepth, m_config.currentRRSMethod->rrDepth,
                                        ETransportMode::EImportance, m_pool);
                Assert(emitterPath.length() > 0 && "Can emitter sample fail?");

                /// Add costs of tracing the new lightpath
                output.cost += (emitterPath.length() - 1) * COST_BSDF;
                importanceWeights[0] = Spectrum(1.0f);
                for (size_t i = 1; i < emitterPath.vertexCount(); ++i)
                    importanceWeights[i] = importanceWeights[i-1] *
                        emitterPath.vertex(i-1)->weight[EImportance] *
                        emitterPath.vertex(i-1)->rrWeight *
                        emitterPath.edge(i-1)->weight[EImportance];
            }

            /*** | t = 1 | absoluteWeight should be v0 * e0 | ***/
            if (t >= 1) {
                Assert(splitFactors.size() == 2);
                /* Set all input parameters used by evaluate */
                Spectrum tmpAbsWeight { v0->weight[ERadiance] * e0.weight[ERadiance] };
                Spectrum tmpNormWeight { tmpAbsWeight };
                if (!m_config.currentRRSMethod->useAbsoluteThroughput)
                    tmpNormWeight /= m_pixelEstimate + Spectrum { 1e-2 };

                /* Evaluate light Image connections and add them to our output */
                std::tie(LrEstimateBDPT, LrCostBDPT) = bdptEvaluate(
                    sensorPath, tmpAbsWeight, tmpNormWeight, importanceWeights,
                    initialSamplePos, emitterPath, result, NOSELF
                );

                Assert(LrEstimateBDPT.isZero() && "Light image is added separately");
                output.cost += LrCostBDPT;

                /* Drop e0 and v1 */
                sensorPath.dropLast();
                splitFactors.pop_back();
            }

            /*** | t = 0 | absoluteWeight should be Spectrum(1.0f) | ***/
            {
                /* Set all input parameters used by evaluate */
                Spectrum tmpAbsWeight { 1.0 },
                         tmpNormWeight { 1.0 };
                if (!m_config.currentRRSMethod->useAbsoluteThroughput)
                    tmpNormWeight /= m_pixelEstimate + Spectrum { 1e-2 };

                /* Evaluate light Image connections and add them to our output */
                std::tie(LrEstimateBDPT, LrCostBDPT) = bdptEvaluate(
                    sensorPath, tmpAbsWeight, tmpNormWeight, importanceWeights,
                    initialSamplePos, emitterPath, result, NOSELF
                );

                Assert(LrEstimateBDPT.isZero() && "Light image is added separately");
                output.cost += LrCostBDPT;
                splitFactors.pop_back();
            }

        } else {
            /* Drop e0, e1, v1, v2 since they are not allocated via pool and cannot be released */
            if (t >= 1) {
                sensorPath.dropLast();
                splitFactors.pop_back();
            }
            splitFactors.pop_back();
        }

        /* Only put a sample into the result if sensor sampling was successful.
           We do not even have a pixel position otherwise. */
        if (t == 1) // For AOVs, they are identical to EARS like this. Other than that, won't change a thing
            output.markAsLeaf(2);
        if (t >= 2)
            result->putSample(initialSamplePos, absoluteWeight * output.totalContribution());
#ifdef BMARS_DEBUG_FP
        disableFPExceptions();
#endif
        /* Compute and register blockstatistics */
        const Spectrum pixelContribution = (absoluteWeight / metricNorm) * output.totalContribution();
        const Spectrum diff = (pixelContribution - expectedContribution);

        /* Sample sensor traces a ray already. Add its cost to global cost */
        output.cost += COST_BSDF;
        /* EmitterPath is always sampled if not resampled every split. Add tracing it to the global cost */
        if (m_config.currentRRSMethod->bmarsReuseLP)
            output.cost += COST_BSDF * (emitterPath.edgeCount() - 1);
        blockStatistics += OutlierRejectedAverage::Sample {
            diff * diff,
            output.cost
        };

#ifdef MARS_INCLUDE_AOVS
        stats.pixelEstimate.add(m_pixelEstimate);
        stats.avgPathLength.add(output.averagePathLength()-1);
        stats.numPaths.add(output.numberOfPaths());
        stats.cost.add(1e+6 * output.cost);
#endif

        /* Make sure only the initial vertex from path initialization remains */
        Assert(sensorPath.edgeCount() == 0 && sensorPath.vertexCount() == 1);
        Assert(splitFactors.empty());
        sensorPath.release(m_pool);

        return { initialSamplePos, output };
    }

    ref<WorkProcessor> clone() const {
        return new BMARSRenderer(m_config);
    }

    MTS_DECLARE_CLASS()
private:
    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    mutable ref<Sampler> m_sampler;
    mutable MemoryPool m_pool;
    ref<ReconstructionFilter> m_rfilter;
    BMARSConfiguration m_config;
    HilbertCurve2D<uint8_t> m_hilbertCurve;
};


/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

BMARSProcess::BMARSProcess(const RenderJob *parent, RenderQueue *queue,
        const BMARSConfiguration &config, BMARSWorkResult *liWR) :
    BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) {
    /// Progress is tracked from the outside.
    this->disableProgress();
    m_result = liWR;
    Assert(!m_config.lightImage || liWR != nullptr);
}

ref<WorkProcessor> BMARSProcess::createWorkProcessor() const {
    return new BMARSRenderer(m_config);
}

void BMARSProcess::develop(int spp) {
    if (!m_config.lightImage)
        return;
    LockGuard lock(m_resultMutex);
    const ImageBlock *lightImage = m_result->getLightImage();
    m_film->setBitmap(m_result->getImageBlock()->getBitmap());
    m_film->addBitmap(lightImage->getBitmap(), 1.0f / spp);
    m_queue->signalRefresh(m_parent);
}

void BMARSProcess::processResult(const WorkResult *wr, bool cancelled) {
    if (cancelled)
        return;
    const BMARSWorkResult *result = static_cast<const BMARSWorkResult *>(wr);
    ImageBlock *block = const_cast<ImageBlock *>(result->getImageBlock());

    LockGuard lock(m_resultMutex);
    if (m_config.lightImage)
        m_result->putBlock(result);
    else
        m_film->put(block);

    m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);
}

void BMARSProcess::bindResource(const std::string &name, int id) {
    BlockedRenderProcess::bindResource(name, id);
}

MTS_IMPLEMENT_CLASS_S(BMARSRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(BMARSProcess, false, BlockedRenderProcess)
MTS_NAMESPACE_END
