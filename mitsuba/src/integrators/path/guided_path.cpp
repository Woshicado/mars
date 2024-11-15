/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob
    Copyright (c) 2017 by ETH Zurich, Thomas Mueller.

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

/// we support outputting several AOVs that can be helpful for research and debugging.
/// since they are computationally expensive, we disable them by default.
/// uncomment the following line to enable outputting AOVs:
// #define MARS_INCLUDE_AOVS

// #define STOCHASTIC_DIV_NUM_SAMPLES
#define LOW_DISCREPANCY_NUM_SAMPLES
#define CLASSIC_RR_TRAIN_DEPTH 5

#include "GitSHA1.h"

#include <OpenImageDenoise/oidn.hpp>

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>

#include "statistics.h"

MTS_NAMESPACE_BEGIN

thread_local StatsImageBlockCache *StatsImageBlockCache::instance = nullptr;
thread_local StatsDescriptorCache *StatsDescriptorCache::instance = nullptr;
thread_local StatsValuesCache *StatsValuesCache::instance = nullptr;

constexpr int WO_HIST_RES = 4;
constexpr int ESTIMATOR_BINS = 6 + WO_HIST_RES * WO_HIST_RES;

constexpr Float COST_BSDF    = 0.3e-7 * 1.7;
constexpr Float COST_GUIDING = 0.3e-7 * 1.7;
constexpr Float COST_NEE     = 0.3e-7 * 1.7;
constexpr Float COST_PRIMARY = 0.3e-7 * 1.7;

constexpr Float OUTLIER_MAX = 50.f;
// constexpr Float OUTLIER_MAX = -1;

static const char* errsToTxt[] = {
    "NoRRS",
    "AlbedoRR",
    "Albedo2RRS",
    "ClassicRRS",
    "ADRRS",
    "ADRRS2",
    "MARS"
};
const static char* splitConfigToText[] = {
    "BsdfNeeShared",
    "BsdfNeeSplit",
    "BsdfNeeSplitSF",
    "BsdfNeeGuidingSplit",
    "BsdfNeeGuidingSplitSF",
    "BsdfNeeGuidingOneSampleSplit",
};

class BlobReader {
public:
    BlobReader(const std::string& filename) : f(filename, std::ios::in | std::ios::binary) {}

    std::string readString() {
        uint32_t length;
        f.read(reinterpret_cast<char*>(&length), sizeof(length));

        std::string result;
        result.resize(length);
        f.read((char *)result.data(), length);

        return result;
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobReader&>::type
    operator >> (Type& Element) {
        uint16_t size;
        Read(&size, 1);
        if (size != sizeof(Type)) {
            SLog(EError, "Tried to read element of size %d but got %d", (int)sizeof(Type), (int)size);
        }

        Read(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Read(T* Dest, size_t Size) {
        f.read(reinterpret_cast<char*>(Dest), Size * sizeof(T));
    }

    bool isValid() const {
        return (bool)(f);
    }

private:
    std::ifstream f;
};

class BlobWriter {
public:
    BlobWriter(const std::string& filename)
        : f(filename, std::ios::out | std::ios::binary) {
    }

    void writeString(const std::string& str) {
        uint32_t length = str.length();
        f.write(reinterpret_cast<const char*>(&length), sizeof(length));
        f.write(str.c_str(), length);
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter&>::type
        operator << (Type Element) {
        uint16_t size = sizeof(Type);
        Write(&size, 1);

        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Write(T* Src, size_t Size) {
        f.write(reinterpret_cast<const char*>(Src), Size * sizeof(T));
    }

private:
    std::ofstream f;
};

class AtomicFloat {
public:
    AtomicFloat() : value(0) {}
    AtomicFloat(Float v) : value(v) {}
    AtomicFloat(const AtomicFloat &other) {
        *this = other;
    }

    AtomicFloat &operator=(Float v) {
        value = v;
        return *this;
    }

    AtomicFloat &operator=(const AtomicFloat &other) {
        value = Float(other.value);
        return *this;
    }

    AtomicFloat &operator+=(Float v) {
        atomicAdd(&value, v);
        return *this;
    }

    operator Float() const { return value; }

private:
    Float value;
};

class AtomicDouble {
public:
    AtomicDouble() : value(0) {}
    AtomicDouble(double v) : value(v) {}
    AtomicDouble(const AtomicDouble &other) {
        *this = other;
    }

    AtomicDouble &operator=(double v) {
        value = v;
        return *this;
    }

    AtomicDouble &operator=(const AtomicDouble &other) {
        value = double(other.value);
        return *this;
    }

    AtomicDouble &operator+=(double v) {
        atomicAdd(&value, v);
        return *this;
    }

    operator double() const { return value; }

private:
    double value;
};

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

inline Float logistic(Float x) {
    return 1 / (1 + std::exp(-x));
}

// Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer {
public:
    AdamOptimizer(Float learningRate, int batchSize = 1, Float epsilon = 1e-08f, Float beta1 = 0.9f, Float beta2 = 0.999f) {
		m_hparams = { learningRate, batchSize, epsilon, beta1, beta2 };
	}

    AdamOptimizer& operator=(const AdamOptimizer& arg) {
        m_state = arg.m_state;
        m_hparams = arg.m_hparams;
        return *this;
    }

    AdamOptimizer(const AdamOptimizer& arg) {
        *this = arg;
    }

    void append(Float gradient, Float statisticalWeight) {
        m_state.batchGradient += gradient * statisticalWeight;
        m_state.batchAccumulation += statisticalWeight;

        if (m_state.batchAccumulation > m_hparams.batchSize) {
            step(m_state.batchGradient / m_state.batchAccumulation);

            m_state.batchGradient = 0;
            m_state.batchAccumulation = 0;
        }
    }

    void step(Float gradient) {
        ++m_state.iter;

        Float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
        m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
        m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);

        // Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
        // since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
        // in *extremely* small and large results that are pretty much never necessary in practice.
        m_state.variable = std::min<Float>(std::max<Float>(m_state.variable, -20.0), 20.0);
    }

    Float variable() const {
        return m_state.variable;
    }

    void read(BlobReader& blob) {
        blob
            >> m_state
            >> m_hparams;
    }

    void dump(BlobWriter& blob) const {
        blob
            << m_state
            << m_hparams;
    }

private:
    struct State {
        int iter = 0;
        Float firstMoment = 0;
        Float secondMoment = 0;
        Float variable = 0;

        Float batchAccumulation = 0;
        Float batchGradient = 0;
    } m_state;

    struct Hyperparameters {
        Float learningRate;
        int batchSize;
        Float epsilon;
        Float beta1;
        Float beta2;
    } m_hparams;
};

enum class ESampleCombination {
    EDiscard,
    EDiscardWithAutomaticBudget,
    EInverseVariance,
};

enum class EBsdfSamplingFractionLoss {
    ENone,
    EKL,
    EVariance,
    EImageKL,
    EImageVariance,
};

enum class ESpatialFilter {
    ENearest,
    EStochasticBox,
    EBox,
};

enum class EDirectionalFilter {
    ENearest,
    EBox,
};

class QuadTreeNode {
public:
    QuadTreeNode() {
        m_children = {};
        for (size_t i = 0; i < m_sum.size(); ++i) {
            m_sum[i] = 0;
        }
    }

    void setSum(int index, Float val) {
        m_sum[index] = val;
    }

    Float sum(int index) const {
        return m_sum[index];
    }

    void copyFrom(const QuadTreeNode& arg) {
        for (int i = 0; i < 4; ++i) {
            setSum(i, arg.sum(i));
            m_children[i] = arg.m_children[i];
        }
    }

    QuadTreeNode(const QuadTreeNode& arg) {
        copyFrom(arg);
    }

    QuadTreeNode& operator=(const QuadTreeNode& arg) {
        copyFrom(arg);
        return *this;
    }

    void setChild(int idx, uint16_t val) {
        m_children[idx] = val;
    }

    uint16_t child(int idx) const {
        return m_children[idx];
    }

    void setSum(Float val) {
        for (int i = 0; i < 4; ++i) {
            setSum(i, val);
        }
    }

    int childIndex(Point2& p) const {
        int res = 0;
        for (int i = 0; i < Point2::dim; ++i) {
            if (p[i] < 0.5f) {
                p[i] *= 2;
            } else {
                p[i] = (p[i] - 0.5f) * 2;
                res |= 1 << i;
            }
        }

        return res;
    }

    // Evaluates the directional irradiance *sum density* (i.e. sum / area) at a given location p.
    // To obtain radiance, the sum density (result of this function) must be divided
    // by the total statistical weight of the estimates that were summed up.
    Float eval(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 4 * sum(index);
        } else {
            return 4 * nodes[child(index)].eval(p, nodes);
        }
    }

    Float pdf(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (!(sum(index) > 0)) {
            return 0;
        }

        const Float factor = 4 * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
        if (isLeaf(index)) {
            return factor;
        } else {
            return factor * nodes[child(index)].pdf(p, nodes);
        }
    }

    int depthAt(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 1;
        } else {
            return 1 + nodes[child(index)].depthAt(p, nodes);
        }
    }

    Point2 sample(Sampler* sampler, const std::vector<QuadTreeNode>& nodes) const {
        int index = 0;

        Float topLeft = sum(0);
        Float topRight = sum(1);
        Float partial = topLeft + sum(2);
        Float total = partial + topRight + sum(3);

        // Should only happen when there are numerical instabilities.
        if (!(total > 0.0f)) {
            return sampler->next2D();
        }

        Float boundary = partial / total;
        Point2 origin = Point2{0.0f, 0.0f};

        Float sample = sampler->next1D();

        if (sample < boundary) {
            SAssert(partial > 0);
            sample /= boundary;
            boundary = topLeft / partial;
        } else {
            partial = total - partial;
            SAssert(partial > 0);
            origin.x = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            boundary = topRight / partial;
            index |= 1 << 0;
        }

        if (sample < boundary) {
            sample /= boundary;
        } else {
            origin.y = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            index |= 1 << 1;
        }

        if (isLeaf(index)) {
            return origin + Float(0.5) * sampler->next2D();
        } else {
            return origin + Float(0.5) * nodes[child(index)].sample(sampler, nodes);
        }
    }

    void record(Point2& p, Float irradiance, std::vector<QuadTreeNode>& nodes) {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        int index = childIndex(p);

        if (isLeaf(index)) {
            m_sum[index] += irradiance;
        } else {
            nodes[child(index)].record(p, irradiance, nodes);
        }
    }

    Float computeOverlappingArea(const Point2& min1, const Point2& max1, const Point2& min2, const Point2& max2) {
        Float lengths[2];
        for (int i = 0; i < 2; ++i) {
            lengths[i] = std::max<Float>(std::min<Float>(max1[i], max2[i]) - std::max<Float>(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1];
    }

    void record(const Point2& origin, Float size, Point2 nodeOrigin, Float nodeSize, Float value, std::vector<QuadTreeNode>& nodes) {
        Float childSize = nodeSize / 2;
        for (int i = 0; i < 4; ++i) {
            Point2 childOrigin = nodeOrigin;
            if (i & 1) { childOrigin[0] += childSize; }
            if (i & 2) { childOrigin[1] += childSize; }

            Float w = computeOverlappingArea(origin, origin + Point2(size), childOrigin, childOrigin + Point2(childSize));
            if (w > 0.0f) {
                if (isLeaf(i)) {
                    m_sum[i] += value * w;
                } else {
                    nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes);
                }
            }
        }
    }

    bool isLeaf(int index) const {
        return child(index) == 0;
    }

    // Ensure that each quadtree node's sum of irradiance estimates
    // equals that of all its children.
    void build(std::vector<QuadTreeNode>& nodes) {
        for (int i = 0; i < 4; ++i) {
            // During sampling, all irradiance estimates are accumulated in
            // the leaves, so the leaves are built by definition.
            if (isLeaf(i)) {
                continue;
            }

            QuadTreeNode& c = nodes[child(i)];

            // Recursively build each child such that their sum becomes valid...
            c.build(nodes);

            // ...then sum up the children's sums.
            Float sum = 0;
            for (int j = 0; j < 4; ++j) {
                if (c.sum(j) < 0)
                    c.setSum(j, 0);
                sum += c.sum(j);
            }
            setSum(i, sum);
        }
    }

private:
    std::array<AtomicFloat, 4> m_sum;
    std::array<uint16_t, 4> m_children;
};


class DTree {
public:
    DTree() {
        m_maxDepth = 0;
        m_nodes.emplace_back();
        m_nodes.front().setSum(0.0f);
    }

    QuadTreeNode& node(size_t i) {
        return m_nodes[i];
    }

    const QuadTreeNode& node(size_t i) const {
        return m_nodes[i];
    }

    Float mean() const {
        if (m_atomic.statisticalWeight == 0) {
            return 0;
        }
        const Float factor = 1 / (M_PI * 4 * m_atomic.statisticalWeight);
        return factor * m_atomic.sum;
    }

    void setMean(const Float& mean) {
        const Float factor = M_PI * 4;
        m_atomic.sum = factor * mean;
    }

    void recordIrradiance(Point2 p, Float irradiance, Float statisticalWeight, EDirectionalFilter directionalFilter) {
        if (std::isfinite(statisticalWeight) && statisticalWeight > 0) {
            m_atomic.statisticalWeight += statisticalWeight;

            if (std::isfinite(irradiance) && irradiance > 0) {
                if (directionalFilter == EDirectionalFilter::ENearest) {
                    m_nodes[0].record(p, irradiance * statisticalWeight, m_nodes);
                } else {
                    int depth = depthAt(p);
                    Float size = std::pow(0.5f, depth);

                    Point2 origin = p;
                    origin.x -= size / 2;
                    origin.y -= size / 2;
                    m_nodes[0].record(origin, size, Point2(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), m_nodes);
                }
            }
        }
    }

    Float pdf(Point2 p) const {
        if (!(mean() > 0)) {
            return 1 / (4 * M_PI);
        }

        return m_nodes[0].pdf(p, m_nodes) / (4 * M_PI);
    }

    int depthAt(Point2 p) const {
        return m_nodes[0].depthAt(p, m_nodes);
    }

    int depth() const {
        return m_maxDepth;
    }

    Point2 sample(Sampler* sampler) const {
        if (!(mean() > 0)) {
            return sampler->next2D();
        }

        Point2 res = m_nodes[0].sample(sampler, m_nodes);

        res.x = math::clamp<Float>(res.x, 0.0, 1.0);
        res.y = math::clamp<Float>(res.y, 0.0, 1.0);

        return res;
    }

    void setNumNodes(size_t numNodes) {
        m_nodes.resize(numNodes);
    }

    void resetSum() {
        for (auto& node : m_nodes) {
            node.setSum(0.f);
        }
    }

    size_t numNodes() const {
        return m_nodes.size();
    }

    Float statisticalWeight() const {
        return m_atomic.statisticalWeight;
    }

    void setStatisticalWeight(Float statisticalWeight) {
        m_atomic.statisticalWeight = statisticalWeight;
    }

    void reset(const DTree& previousDTree, int newMaxDepth, Float subdivisionThreshold) {
        m_atomic = Atomic{};
        m_maxDepth = 0;
        m_nodes.clear();
        m_nodes.emplace_back();

        struct StackNode {
            size_t nodeIndex;
            size_t otherNodeIndex;
            const DTree* otherDTree;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 0, &previousDTree, 1});

        const Float total = previousDTree.m_atomic.sum;

        // Create the topology of the new DTree to be the refined version
        // of the previous DTree. Subdivision is recursive if enough energy is there.
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i) {
                const QuadTreeNode& otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
                const Float fraction = total > Epsilon  ? (otherNode.sum(i) / total) : std::pow(0.25f, sNode.depth);
                SAssert(fraction <= 1.0f + Epsilon);

                if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
                    if (!otherNode.isLeaf(i)) {
                        SAssert(sNode.otherDTree == &previousDTree);
                        nodeIndices.push({m_nodes.size(), otherNode.child(i), &previousDTree, sNode.depth + 1});
                    } else {
                        nodeIndices.push({m_nodes.size(), m_nodes.size(), this, sNode.depth + 1});
                    }

                    m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
                    auto ssum = otherNode.sum(i) / 4;
                    m_nodes.emplace_back();
                    m_nodes.back().setSum(ssum);

                    if (m_nodes.size() > std::numeric_limits<uint16_t>::max()) {
                        SLog(EWarn, "DTreeWrapper hit maximum children count.");
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();

        for (auto& node : m_nodes) {
            node.setSum(0);
        }
    }

    size_t approxMemoryFootprint() const {
        return m_nodes.capacity() * sizeof(QuadTreeNode) + sizeof(*this);
    }

    void build() {
        auto& root = m_nodes[0];

        // Build the quadtree recursively, starting from its root.
        root.build(m_nodes);

        // Ensure that the overall sum of irradiance estimates equals
        // the sum of irradiance estimates found in the quadtree.
        Float sum = 0;
        for (int i = 0; i < 4; ++i) {
            sum += root.sum(i);
        }
        m_atomic.sum = sum;
    }

private:
    std::vector<QuadTreeNode> m_nodes;

    struct Atomic {
        AtomicFloat sum;
        AtomicFloat statisticalWeight;
    } m_atomic;

public:
    int m_maxDepth;
};
template<typename FloatType>
struct EstimatorStatistics {
    void atomicAddSpec(Spectrum *dest, const Spectrum &src) {
    for (int c = 0; c < SPECTRUM_SAMPLES; ++c)
        atomicAdd(&(*dest)[c], src[c]);
    }

    template<typename>
    friend struct EstimatorStatistics;
    void addSample(Float weight, Spectrum value, Float cost) {
        weight = 1;//weight * weight;

        atomicAddSpec (&firstMoment , weight * value);
        atomicAddSpec (&secondMoment, weight * value * value);
        this->cost         += weight * cost;
        this->weight       += weight;
    }

    template<typename OtherFloatType>
    EstimatorStatistics<FloatType> &operator+=(const EstimatorStatistics<OtherFloatType> &other) {
        atomicAddSpec (&firstMoment , other.firstMoment);
        atomicAddSpec (&secondMoment, other.secondMoment);
        atomicAdd( &lrSamples, other.lrSamples);
        atomicAdd( &guidingSamples, other.guidingSamples);
        atomicAdd( &neeSamples, other.neeSamples);
        atomicAdd( &samplesTaken, other.samplesTaken);
        cost         += other.cost;
        weight       += other.weight;
        return *this;
    }

    void addSplitStatistics(const int lr, const int guiding, const int nee) {
        samplesTaken += 1;
        lrSamples += lr;
        guidingSamples += guiding;
        neeSamples += nee;
    }

    void reset() {
        *this = EstimatorStatistics<FloatType>();
    }

    Float getCost() const {
        return weight != 0 ? cost / weight : 0;
    }

    Spectrum getFirstMoment() const {
        return weight != 0 ? firstMoment / weight : Spectrum { 0.0 };
    }

    Spectrum getSecondMoment() const {
        return weight != 0 ? secondMoment / weight : Spectrum { 0.0 };
    }

    Spectrum getSquareError() const {
        if (weight < 16)
            // not enough data available
            return Spectrum { 0.0 };
        return secondMoment / weight;
    }

    Spectrum computeVariance() const {
        if (weight < 16)
            // not enough data available
            return Spectrum { 0.0 };

        Spectrum result;
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
            result[i] = std::max<Float>(
                            secondMoment[i] / weight - Float(std::pow(firstMoment[i] / weight, 2)),
                            Float(0)
                        );
        return result;
    }

    Float getMean() const {
        return weight != 0 ? firstMoment / weight : 0;
    }

    Float getWeight() const {
        return weight;
    }

    Float getAverageLR() {
        return Float(lrSamples) / samplesTaken;
    }
    Float getAverageNEE() {
        return Float(neeSamples) / samplesTaken;
    }
    Float getAverageGuiding() {
        return Float(guidingSamples) / samplesTaken;
    }

private:
    int lrSamples { 0 };
    int guidingSamples { 0 };
    int neeSamples { 0 };
    int samplesTaken { 0 };
    Spectrum firstMoment   { 0.f };
    Spectrum secondMoment  { 0.f };
    FloatType weight       { 0.f };
    FloatType cost         { 0.f };
};

struct DTreeRecord {
    Vector d;
    Float radiance, product, imageContribution;
    Float woPdf, bsdfPdf, dTreePdf;
    Float statisticalWeight;
    bool isDelta;
};

struct DTreeWrapper {
public:
    DTreeWrapper() {
    }

    void record(const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
        if (!rec.isDelta) {
            Float irradiance = rec.radiance / rec.woPdf;
            building.recordIrradiance(dirToCanonical(rec.d), irradiance, rec.statisticalWeight, directionalFilter);
        }

        if (bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && rec.product > 0) {
            const bool isKL =
                bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ||
                bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EImageKL;
            const bool isImage =
                bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EImageKL ||
                bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EImageVariance;
            optimizeBsdfSamplingFraction(
                rec,
                isKL ? 1.0f : 2.0f,
                isImage ? rec.imageContribution : rec.product
            );
        }
    }

    static Vector canonicalToDir(Point2 p) {
        const Float cosTheta = 2 * p.x - 1;
        const Float phi = 2 * M_PI * p.y;

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        math::sincos(phi, &sinPhi, &cosPhi);

        return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
    }

    static Point2 dirToCanonical(const Vector& d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return {0, 0};
        }

        const Float cosTheta = std::min<Float>(std::max<Float>(d.z, -1.0), 1.0);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2.0 * M_PI;

        return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
    }

    void build() {
        building.build();
        sampling = building;

        for (int index = 0; index < ESTIMATOR_BINS; ++index) {
            if (std::max(bsdfStatistics[index].getWeight(),
                         guidingStatistics[index].getWeight()) >= 16) {
                primaryCostBsdf[index] = bsdfStatistics[index].getCost();
                primaryCostNee[index] = neeStatistics[index].getCost();
                primaryCostGuiding[index] = guidingStatistics[index].getCost();
                primaryCost[index] = primaryCostBsdf[index] + primaryCostNee[index] + primaryCostGuiding[index];
                primary2ndMomBsdf[index] = bsdfStatistics[index].getSecondMoment();
                primary2ndMomNee[index] = neeStatistics[index].getSecondMoment();
                primary2ndMomGuiding[index] = guidingStatistics[index].getSecondMoment();
                primary2ndMom[index] =
                    primary2ndMomBsdf[index]
                    + primary2ndMomNee[index]
                    + primary2ndMomGuiding[index]
                    + 2 * bsdfStatistics[index].getFirstMoment() * neeStatistics[index].getFirstMoment()
                    + 2 * bsdfStatistics[index].getFirstMoment() * guidingStatistics[index].getFirstMoment()
                    + 2 * neeStatistics[index].getFirstMoment() * guidingStatistics[index].getFirstMoment();
                primary1stMomBG[index] = bsdfStatistics[index].getFirstMoment() + guidingStatistics[index].getFirstMoment();

                primaryVarBsdf[index]    = bsdfStatistics[index].computeVariance();
                primaryVarNee[index]     = neeStatistics[index].computeVariance();
                primaryVarGuiding[index] = guidingStatistics[index].computeVariance();

                primaryRelVar[index] = 0;
            } else {
                primaryCost[index] = 0;
                primaryCostBsdf[index] = 0;
                primaryCostNee[index] = 0;
                primaryCostGuiding[index] = 0;
                primary2ndMomBsdf[index] = Spectrum { 0.0 };
                primary2ndMomNee[index] = Spectrum { 0.0 };
                primary2ndMomGuiding[index] = Spectrum { 0.0 };
                primary2ndMom[index] = Spectrum { 0.0 };
                primaryVarBsdf[index] = Spectrum { 0.0 };
                primaryVarNee[index] = Spectrum { 0.0 };
                primaryVarGuiding[index] = Spectrum { 0.0 };
                primaryRelVar[index] = 0;
            }

            neeStatistics[index].reset();
            bsdfStatistics[index].reset();
            guidingStatistics[index].reset();

            if (irradianceStatistics[index].hasSufficientSamples()) {
                irradianceEstimate[index] = irradianceStatistics[index].getIrradianceEstimate();
                irradianceWeight[index] = irradianceStatistics[index].getWeight();
            } else {
                irradianceEstimate[index] = Spectrum { 0.f };
                irradianceWeight[index] = 0;
            }

            irradianceStatistics[index].reset();
        }

        if (indirectIrradianceStatistics.hasSufficientSamples()) {
            indirectIrradianceEstimate = indirectIrradianceStatistics.getIrradianceEstimate();
        } else {
            indirectIrradianceEstimate = Spectrum { 0.f };
        }

        indirectIrradianceStatistics.reset();
    }

    void reset(int maxDepth, Float subdivisionThreshold) {
        building.reset(sampling, maxDepth, subdivisionThreshold);
    }

    bool hasPrimaryEstimates(int index) const {
        return !primary2ndMom[index].isZero();
    }

    int mapContextToIndex(const Vector3 &normal, const Vector3 &wo, Float roughness, bool canUseWoHistogram) const {
        if (canUseWoHistogram && roughness < 0.25) {
            const Point2 p = dirToCanonical(wo);
            const int result =
                std::min(int(p.x * WO_HIST_RES), WO_HIST_RES-1) +
                std::min(int(p.y * WO_HIST_RES), WO_HIST_RES-1) * WO_HIST_RES;
            return result + 6;
        }

        Float max = 0;
        int result = 0;
        for (int dim = 0; dim < 6; ++dim) {
            const Float sign = 1 - 2 * (dim % 2);
            const Float v = sign * normal[dim / 2];

            if (v > max) {
                max = v;
                result = dim;
            }
        }
        return result;
    }

    Spectrum getPrimary1stMomBG(int index) const {
        return primary1stMomBG[index];
    }

    Spectrum getPrimary2ndMom(int index) const {
        return primary2ndMom[index];
    }

    Spectrum getPrimary2ndMomBsdf(int index) const {
        return primary2ndMomBsdf[index];
    }

    Spectrum getPrimary2ndMomNee(int index) const {
        return primary2ndMomNee[index];
    }

    Spectrum getPrimary2ndMomGuiding(int index) const {
        return primary2ndMomGuiding[index];
    }

    Spectrum getPrimaryVariance(int index) const {
        return primaryVarBsdf[index] + primaryVarNee[index] + primaryVarGuiding[index];
    }

    Spectrum getPrimaryVarianceBsdf(int index) const {
        return primaryVarBsdf[index];
    }

    Spectrum getPrimaryVarianceNee(int index) const {
        return primaryVarNee[index];
    }

    Spectrum getPrimaryVarianceGuiding(int index) const {
        return primaryVarGuiding[index];
    }

    Float getPrimaryCost(int index) const {
        return primaryCost[index];
    }

    Float getPrimaryCostBsdf(int index) const {
        return primaryCostBsdf[index];
    }

    Float getPrimaryCostNee(int index) const {
        return primaryCostNee[index];
    }

    Float getPrimaryCostGuiding(int index) const {
        return primaryCostGuiding[index];
    }

    Float getIrradianceWeight(int index) const {
        return irradianceWeight[index];
    }

    const Spectrum &getIrradianceEstimate(int index) const {
        return irradianceEstimate[index];
    }

    const Spectrum &getIndirectIrradianceEstimate() const {
        return indirectIrradianceEstimate;
    }

    Vector sample(Sampler* sampler) const {
        return canonicalToDir(sampling.sample(sampler));
    }

    Float pdf(const Vector& dir) const {
        return sampling.pdf(dirToCanonical(dir));
    }

    Float diff(const DTreeWrapper& other) const {
        return 0.0f;
    }

    int depth() const {
        return sampling.depth();
    }

    size_t numNodes() const {
        return sampling.numNodes();
    }

    Float meanRadiance() const {
        return sampling.mean();
    }

    Float statisticalWeight() const {
        return sampling.statisticalWeight();
    }

    Float statisticalWeightBuilding() const {
        return building.statisticalWeight();
    }

    void setStatisticalWeightBuilding(Float statisticalWeight) {
        building.setStatisticalWeight(statisticalWeight);
    }

    size_t approxMemoryFootprint() const {
        return building.approxMemoryFootprint() + sampling.approxMemoryFootprint();
    }

    inline Float bsdfSamplingFraction(Float variable) const {
        return logistic(variable);
    }

    inline Float dBsdfSamplingFraction_dVariable(Float variable) const {
        Float fraction = bsdfSamplingFraction(variable);
        return fraction * (1 - fraction);
    }

    inline Float bsdfSamplingFraction() const {
        return bsdfSamplingFraction(bsdfSamplingFractionOptimizer.variable());
    }

    void optimizeBsdfSamplingFraction(const DTreeRecord& rec, Float ratioPower, Float contribution) {
        m_lock.lock();

        // GRADIENT COMPUTATION
        Float variable = bsdfSamplingFractionOptimizer.variable();
        Float samplingFraction = bsdfSamplingFraction(variable);

        // Loss gradient w.r.t. sampling fraction
        Float mixPdf = samplingFraction * rec.bsdfPdf + (1 - samplingFraction) * rec.dTreePdf;
        Float ratio = std::pow(contribution / mixPdf, ratioPower);
        Float dLoss_dSamplingFraction = -ratio / rec.woPdf * (rec.bsdfPdf - rec.dTreePdf);

        // Chain rule to get loss gradient w.r.t. trainable variable
        Float dLoss_dVariable = dLoss_dSamplingFraction * dBsdfSamplingFraction_dVariable(variable);

        // We want some regularization such that our parameter does not become too big.
        // We use l2 regularization, resulting in the following linear gradient.
        Float l2RegGradient = 0.01f * variable;

        Float lossGradient = l2RegGradient + dLoss_dVariable;

        // ADAM GRADIENT DESCENT
        bsdfSamplingFractionOptimizer.append(lossGradient, rec.statisticalWeight);

        m_lock.unlock();
    }

    void read(BlobReader& blob) {
        bsdfSamplingFractionOptimizer.read(blob);

        Float mean;
        Float statisticalWeight;
        uint64_t numNodes;

        blob
            >> (Float&)mean
            >> (Float&)statisticalWeight
            >> (uint64_t&)numNodes
            >> (int&)sampling.m_maxDepth;

        for (int index = 0; index < ESTIMATOR_BINS; ++index) {
            blob
                >> (Float&)irradianceWeight[index]
                >> (Float&)primaryCost[index]
                >> (Float&)primaryCostBsdf[index]
                >> (Float&)primaryCostNee[index]
                >> (Float&)primaryCostGuiding[index]
            ;

            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)irradianceEstimate[index][i];

            /// Second Moment
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primary2ndMomNee[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primary2ndMomBsdf[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primary2ndMomGuiding[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primary2ndMom[index][i];

            /// Variance
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primaryVarBsdf[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primaryVarNee[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob >> (Float&)primaryVarGuiding[index][i];

        }

        for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
            blob >> (Float&)indirectIrradianceEstimate[i];

        sampling.setNumNodes(numNodes);
        sampling.setStatisticalWeight(statisticalWeight);
        sampling.setMean(mean);

        for (size_t i = 0; i < sampling.numNodes(); ++i) {
            auto& node = sampling.node(i);
            for (int j = 0; j < 4; ++j) {
                Float mean;
                uint16_t child;

                blob >> (Float&)mean;
                mean = std::max<Float>(mean, 0);
                //SAssert(mean >= 0.0f);

                blob >> (uint16_t&)child;

                node.setSum(j, mean);
                node.setChild(j, child);
            }
        }

        building = sampling;
        building.resetSum();
    }

    void dump(BlobWriter& blob) const {
        // bsdfSamplingFractionOptimizer.dump(blob);

        blob
            << (Float)sampling.mean()
            << (Float)sampling.statisticalWeight()
            << (uint64_t)sampling.numNodes()
            << (int)sampling.m_maxDepth;

        for (int index = 0; index < ESTIMATOR_BINS; ++index) {
            blob
                << (Float)irradianceWeight[index]
                << (Float)primaryCost[index]
                << (Float)primaryCostBsdf[index]
                << (Float)primaryCostNee[index]
                << (Float)primaryCostGuiding[index]
            ;

            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)irradianceEstimate[index][i];

            /// Second Moment
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primary2ndMomNee[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primary2ndMomBsdf[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primary2ndMomGuiding[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primary2ndMom[index][i];

            /// Variance
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primaryVarBsdf[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primaryVarNee[index][i];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                blob << (Float)primaryVarGuiding[index][i];

        }

        for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
            blob << (Float)indirectIrradianceEstimate[i];

        for (size_t i = 0; i < sampling.numNodes(); ++i) {
            const auto& node = sampling.node(i);
            for (int j = 0; j < 4; ++j) {
                blob << (Float)node.sum(j) << (uint16_t)node.child(j);
            }
        }
    }

    EstimatorStatistics<AtomicDouble> neeStatistics[ESTIMATOR_BINS],
                                      bsdfStatistics[ESTIMATOR_BINS],
                                      guidingStatistics[ESTIMATOR_BINS];

    struct IrradianceEstimator {
        void reset() {
            *this = IrradianceEstimator();
        }

        void addSample(const Spectrum &value, Float weight) {
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                irradiance[i] += value[i];
            this->weight += weight;
        }

        Spectrum getIrradianceEstimate() const {
            Spectrum result;
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                result[i] = irradiance[i] / weight;
            return result;
        }

        Float getWeight() const {
            return weight;
        }

        bool hasSufficientSamples() const {
            return weight >= 16;
        }

    private:
        std::array<AtomicFloat, SPECTRUM_SAMPLES> irradiance;
        AtomicFloat weight { 0.f };
    } irradianceStatistics[ESTIMATOR_BINS], indirectIrradianceStatistics;

private:
    DTree building;
    DTree sampling;

    Spectrum indirectIrradianceEstimate;
    Spectrum irradianceEstimate[ESTIMATOR_BINS];
    Float irradianceWeight[ESTIMATOR_BINS];

    Spectrum primary1stMomBG[ESTIMATOR_BINS];
    Spectrum primary2ndMom[ESTIMATOR_BINS];
    Spectrum primary2ndMomBsdf[ESTIMATOR_BINS];
    Spectrum primary2ndMomNee[ESTIMATOR_BINS];
    Spectrum primary2ndMomGuiding[ESTIMATOR_BINS];
    Spectrum primaryVarBsdf[ESTIMATOR_BINS];
    Spectrum primaryVarNee[ESTIMATOR_BINS];
    Spectrum primaryVarGuiding[ESTIMATOR_BINS];
    Float primaryCost[ESTIMATOR_BINS];
    Float primaryCostBsdf[ESTIMATOR_BINS];
    Float primaryCostNee[ESTIMATOR_BINS];
    Float primaryCostGuiding[ESTIMATOR_BINS];
    Float primaryRelVar[ESTIMATOR_BINS];

    AdamOptimizer bsdfSamplingFractionOptimizer{0.01f};

    class SpinLock {
    public:
        SpinLock() {
            m_mutex.clear(std::memory_order_release);
        }

        SpinLock(const SpinLock& other) { m_mutex.clear(std::memory_order_release); }
        SpinLock& operator=(const SpinLock& other) { return *this; }

        void lock() {
            while (m_mutex.test_and_set(std::memory_order_acquire)) { }
        }

        void unlock() {
            m_mutex.clear(std::memory_order_release);
        }
    private:
        std::atomic_flag m_mutex;
    } m_lock;
};

struct STreeNode {
    STreeNode() {
        children = {};
        isLeaf = true;
        axis = 0;
    }

    int childIndex(Point& p) const {
        if (p[axis] < 0.5f) {
            p[axis] *= 2;
            return 0;
        } else {
            p[axis] = (p[axis] - 0.5f) * 2;
            return 1;
        }
    }

    int nodeIndex(Point& p) const {
        return children[childIndex(p)];
    }

    DTreeWrapper* dTreeWrapper(Point& p, Vector& size, std::vector<STreeNode>& nodes) {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return &dTree;
        } else {
            size[axis] /= 2;
            return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes);
        }
    }

    const DTreeWrapper* dTreeWrapper() const {
        return &dTree;
    }

    int depth(Point& p, const std::vector<STreeNode>& nodes) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return 1;
        } else {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int depth(const std::vector<STreeNode>& nodes) const {
        int result = 1;

        if (!isLeaf) {
            for (auto c : children) {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void forEachLeaf(
        std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func,
        Point p, Vector size, const std::vector<STreeNode>& nodes) const {

        if (isLeaf) {
            func(&dTree, p, size);
        } else {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i) {
                Point childP = p;
                if (i == 1) {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    Float computeOverlappingVolume(const Point& min1, const Point& max1, const Point& min2, const Point& max2) {
        Float lengths[3];
        for (int i = 0; i < 3; ++i) {
            lengths[i] = std::max<Float>(std::min<Float>(max1[i], max2[i]) - std::max<Float>(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1] * lengths[2];
    }

    void record(const Point& min1, const Point& max1, Point min2, Vector size2, const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, std::vector<STreeNode>& nodes) {
        Float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
        if (w > 0) {
            if (isLeaf) {
                dTree.record({ rec.d, rec.radiance, rec.product, rec.imageContribution, rec.woPdf, rec.bsdfPdf, rec.dTreePdf, rec.statisticalWeight * w, rec.isDelta }, directionalFilter, bsdfSamplingFractionLoss);
            } else {
                size2[axis] /= 2;
                for (int i = 0; i < 2; ++i) {
                    if (i & 1) {
                        min2[axis] += size2[axis];
                    }

                    nodes[children[i]].record(min1, max1, min2, size2, rec, directionalFilter, bsdfSamplingFractionLoss, nodes);
                }
            }
        }
    }

    void read(BlobReader& blob) {
        blob
            >> (bool&)isLeaf
            >> (int&)axis
            >> (uint32_t&)children[0] >> (uint32_t&)children[1];

        if (isLeaf)
            dTree.read(blob);
    }

    void dump(BlobWriter& blob) const {
        blob
            << (bool)isLeaf
            << (int)axis
            << (uint32_t)children[0] << (uint32_t)children[1];

        if (isLeaf)
            dTree.dump(blob);
    }

    bool isLeaf;
    DTreeWrapper dTree;
    int axis;
    std::array<uint32_t, 2> children;
};


class STree {
public:
    STree(const AABB& aabb) {
        clear();

        m_aabb = aabb;

        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Vector size = m_aabb.max - m_aabb.min;
        Float maxSize = std::max(std::max(size.x, size.y), size.z);
        m_aabb.max = m_aabb.min + Vector(maxSize);
    }

    void clear() {
        m_nodes.clear();
        m_nodes.emplace_back();
    }

    void subdivideAll() {
        int nNodes = (int)m_nodes.size();
        for (int i = 0; i < nNodes; ++i) {
            if (m_nodes[i].isLeaf) {
                subdivide(i, m_nodes);
            }
        }
    }

    void subdivide(int nodeIdx, std::vector<STreeNode>& nodes) {
        // Add 2 child nodes
        nodes.resize(nodes.size() + 2);

        if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
            SLog(EWarn, "DTreeWrapper hit maximum children count.");
            return;
        }

        STreeNode& cur = nodes[nodeIdx];
        for (int i = 0; i < 2; ++i) {
            uint32_t idx = (uint32_t)nodes.size() - 2 + i;
            cur.children[i] = idx;
            nodes[idx].axis = (cur.axis + 1) % 3;
            nodes[idx].dTree = cur.dTree;
            nodes[idx].dTree.setStatisticalWeightBuilding(nodes[idx].dTree.statisticalWeightBuilding() / 2);
        }
        cur.isLeaf = false;
        cur.dTree = {}; // Reset to an empty dtree to save memory.
    }

    DTreeWrapper* dTreeWrapper(Point p, Vector& size) {
        size = m_aabb.getExtents();
        p = Point(p - m_aabb.min);
        p.x /= size.x;
        p.y /= size.y;
        p.z /= size.z;

        return m_nodes[0].dTreeWrapper(p, size, m_nodes);
    }

    DTreeWrapper* dTreeWrapper(Point p) {
        Vector size;
        return dTreeWrapper(p, size);
    }

    void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper*)> func) const {
        for (auto& node : m_nodes) {
            if (node.isLeaf) {
                func(&node.dTree);
            }
        }
    }

    void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func) const {
        m_nodes[0].forEachLeaf(func, m_aabb.min, m_aabb.max - m_aabb.min, m_nodes);
    }

    void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper*)> func) {
        int nDTreeWrappers = static_cast<int>(m_nodes.size());

#pragma omp parallel for
        for (int i = 0; i < nDTreeWrappers; ++i) {
            if (m_nodes[i].isLeaf) {
                func(&m_nodes[i].dTree);
            }
        }
    }

    void record(const Point& p, const Vector& dTreeVoxelSize, DTreeRecord rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
        Float volume = 1;
        for (int i = 0; i < 3; ++i) {
            volume *= dTreeVoxelSize[i];
        }

        rec.statisticalWeight /= volume;
        m_nodes[0].record(p - dTreeVoxelSize * 0.5f, p + dTreeVoxelSize * 0.5f, m_aabb.min, m_aabb.getExtents(), rec, directionalFilter, bsdfSamplingFractionLoss, m_nodes);
    }

    void read(BlobReader& blob) {
        size_t numNodes;

        blob
            >> (size_t&)numNodes
            >> (Float&)m_aabb.min.x >> (Float&)m_aabb.min.y >> (Float&)m_aabb.min.z
            >> (Float&)m_aabb.max.x >> (Float&)m_aabb.max.y >> (Float&)m_aabb.max.z;

        m_nodes.resize(numNodes);

        for (size_t i = 0; i < m_nodes.size(); ++i) {
            auto& node = m_nodes[i];
            node.read(blob);
        }
    }

    void dump(BlobWriter& blob) const {
        blob
            << (size_t)m_nodes.size()
            << (Float)m_aabb.min.x << (Float)m_aabb.min.y << (Float)m_aabb.min.z
            << (Float)m_aabb.max.x << (Float)m_aabb.max.y << (Float)m_aabb.max.z;

        for (size_t i = 0; i < m_nodes.size(); ++i) {
            auto& node = m_nodes[i];
            node.dump(blob);
        }
    }

    bool shallSplit(const STreeNode& node, int depth, size_t samplesRequired) {
        return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.statisticalWeightBuilding() > samplesRequired;
    }

    void refine(size_t sTreeThreshold, int maxMB) {
        if (maxMB >= 0) {
            size_t approxMemoryFootprint = 0;
            for (const auto& node : m_nodes) {
                approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
            }

            if (approxMemoryFootprint / 1000000 >= (size_t)maxMB) {
                return;
            }
        }

        struct StackNode {
            size_t index;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0,  1});
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            // Subdivide if needed and leaf
            if (m_nodes[sNode.index].isLeaf) {
                if (shallSplit(m_nodes[sNode.index], sNode.depth, sTreeThreshold)) {
                    subdivide((int)sNode.index, m_nodes);
                }
            }

            // Add children to stack if we're not
            if (!m_nodes[sNode.index].isLeaf) {
                const STreeNode& node = m_nodes[sNode.index];
                for (int i = 0; i < 2; ++i) {
                    nodeIndices.push({node.children[i], sNode.depth + 1});
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();
    }

    const AABB& aabb() const {
        return m_aabb;
    }

private:
    std::vector<STreeNode> m_nodes;
    AABB m_aabb;
};


static StatsCounter avgPathLength("Guided path tracer", "Average path length", EAverage);

class GuidedPathTracer : public MonteCarloIntegrator {
private:
    struct LiInput {
        Spectrum throughput;
        Spectrum relativeThroughput;
        Spectrum liEstimate;
        RayDifferential ray;
        RadianceQueryRecord rRec;
        bool scattered { false };
        Float eta { 1.f };
        Float bsdfPdfCorrection { 1.f };
        int breadth { 1 };
    };

    struct LiOutput {
        Spectrum reflected { 0.f };
        Spectrum emitted { 0.f };
        Float cost { 0.f };
        Float depthAcc { 0.f };
        Float depthWeight { 0.f };
        int lrSamples { 0 };
        int guidingSamples { 0 };
        int neeSamples { 0 };

        Spectrum totalContribution() const {
            return reflected + emitted;
        }

        Float averagePathLength() const {
            return depthWeight > 0 ? depthAcc / depthWeight : 0;
        }

        void markAsLeaf(int depth) {
            depthAcc = depth;
            depthWeight = 1;
        }
    };

    enum MARSSplitConfig {
        BsdfNeeShared,          // 1-sample MIS for guiding bsdf, with shared splitting factor for NEE and BSDF
        BsdfNeeSplit,           // 1-sample MIS for guiding bsdf, just separate splitting factors
        BsdfNeeSplitSF,         // 1-sample MIS for guiding bsdf, separate splitting factors, and include them in MIS weights
        BsdfNeeGuidingSplit,    // Multi-sample MIS with 3 separate splitting factors
        BsdfNeeGuidingSplitSF,  // Multi-sample MIS with 3 separate splitting factors, and include them in MIS weights

        BsdfNeeGuidingOneSampleSplit
    };

public:
    GuidedPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
        m_oidnDevice = oidn::newDevice();
        m_oidnDevice.commit();

        m_neeStr = props.getString("nee", "never");
        if (m_neeStr == "never") {
            m_nee = ENever;
        } else if (m_neeStr == "kickstart") {
            m_nee = EKickstart;
        } else if (m_neeStr == "always") {
            m_nee = EAlways;
        } else {
            Assert(false);
        }

        m_splittingMin = props.getFloat("splittingMin", 0.05f);
        m_splittingMax = props.getFloat("splittingMax", 20);
        m_branchingMax = props.getFloat("branchingMax", std::numeric_limits<Float>::infinity());

        m_rrsStr = props.getString("rrs", "noRR");
        std::string rrsStr = m_rrsStr;
        m_addAdjointEstimate = rrsStr.back() == 'J';
        if (m_addAdjointEstimate)
            rrsStr.pop_back();
        m_useAbsoluteMetric = rrsStr.back() == 'A';
        if (m_useAbsoluteMetric)
            rrsStr.pop_back();

        if (rrsStr.back() != 'S') {
            m_splittingMax = 1.f;
        } else {
            rrsStr.pop_back();
        }

        if (rrsStr == "noRR")      m_rrsMode = ENoRRS; else
        if (rrsStr == "albedoRR")  m_rrsMode = EAlbedoRR; else
        if (rrsStr == "albedo2RR") m_rrsMode = EAlbedo2RRS; else
        if (rrsStr == "classicRR") m_rrsMode = EClassicRRS; else
        if (rrsStr == "ADRR")  m_rrsMode = EADRRS; else
        if (rrsStr == "AD2RR") m_rrsMode = EADRRS2; else
        if (rrsStr == "MAR")   m_rrsMode = EMARS; else
        SAssert(false);

        /* Get MARS specific configuration */
        std::string splitConfigStr = props.getString("splitConfig", "BN");
        if (splitConfigStr == "BN")     { m_splitConfig = BsdfNeeShared; } else
        if (splitConfigStr == "BNS")    { m_splitConfig = BsdfNeeSplit; } else
        if (splitConfigStr == "BNSsf")  { m_splitConfig = BsdfNeeSplitSF; } else
        if (splitConfigStr == "BNGS")   { m_splitConfig = BsdfNeeGuidingSplit; } else
        if (splitConfigStr == "BNGSsf") { m_splitConfig = BsdfNeeGuidingSplitSF; } else
        if (splitConfigStr == "BNGSOS") { m_splitConfig = BsdfNeeGuidingOneSampleSplit; } else
        SAssert(m_rrsMode != EMARS && "SplitConfig has to be set correctly to guide with MARS");

        m_trainingIterations = props.getInteger("trainingIterations", 9);
        m_disableGuiding = props.getBoolean("disableGuiding", false);

        m_sampleCombinationStr = props.getString("sampleCombination", "automatic");
        if (m_sampleCombinationStr == "discard") {
            m_sampleCombination = ESampleCombination::EDiscard;
        } else if (m_sampleCombinationStr == "automatic") {
            m_sampleCombination = ESampleCombination::EDiscardWithAutomaticBudget;
        } else if (m_sampleCombinationStr == "inversevar") {
            m_sampleCombination = ESampleCombination::EInverseVariance;
        } else {
            Assert(false);
        }

        m_spatialFilterStr = props.getString("spatialFilter", "nearest");
        if (m_spatialFilterStr == "nearest") {
            m_spatialFilter = ESpatialFilter::ENearest;
        } else if (m_spatialFilterStr == "stochastic") {
            m_spatialFilter = ESpatialFilter::EStochasticBox;
        } else if (m_spatialFilterStr == "box") {
            m_spatialFilter = ESpatialFilter::EBox;
        } else {
            Assert(false);
        }

        m_directionalFilterStr = props.getString("directionalFilter", "nearest");
        if (m_directionalFilterStr == "nearest") {
            m_directionalFilter = EDirectionalFilter::ENearest;
        } else if (m_directionalFilterStr == "box") {
            m_directionalFilter = EDirectionalFilter::EBox;
        } else {
            Assert(false);
        }

        m_bsdfSamplingFractionLossStr = props.getString("bsdfSamplingFractionLoss", "none");
        if (m_bsdfSamplingFractionLossStr == "none") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
        } else if (m_bsdfSamplingFractionLossStr == "kl") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EKL;
        } else if (m_bsdfSamplingFractionLossStr == "var") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EVariance;
        } else if (m_bsdfSamplingFractionLossStr == "ikl") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EImageKL;
        } else if (m_bsdfSamplingFractionLossStr == "ivar") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EImageVariance;
        } else {
            Assert(false);
        }

        m_sdTreeMaxMemory = props.getInteger("sdTreeMaxMemory", -1);
        m_sTreeThreshold = props.getInteger("sTreeThreshold", 12000);
        m_dTreeThreshold = props.getFloat("dTreeThreshold", 0.01f);
        m_bsdfSamplingFraction = props.getFloat("bsdfSamplingFraction", 0.5f);
        m_sppPerPass = props.getInteger("sppPerPass", 1);

        m_budgetStr = props.getString("budgetType", "seconds");
        if (m_budgetStr == "spp") {
            m_budgetType = ESpp;
        } else if (m_budgetStr == "seconds") {
            m_budgetType = ESeconds;
        } else {
            Assert(false);
        }

        m_budget = props.getFloat("budget", 256.0f);
        m_trainingBudget = props.getFloat("trainingBudget", 120.0f);
        m_renderingBudget = props.getFloat("renderingBudget", 120.0f);
        m_dumpSDTree = props.getBoolean("dumpSDTree", false);
        m_sdTreePath = props.getString("sdTreePath", "");
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        /* This is a sampling-based integrator - parallelize */
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());

        proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    void resetPathHistograms() {
        SAssert(m_maxDepth >= 0);

        m_pathLengthHistogram.resize(m_maxDepth);
        std::fill(m_pathLengthHistogram.begin(), m_pathLengthHistogram.end(), 0);

        m_contributionHistogram.resize(m_maxDepth);
        std::fill(m_contributionHistogram.begin(), m_contributionHistogram.end(), 0);
    }

    void resetSDTree() {
        Log(EInfo, "Resetting distributions for sampling.");

        m_sdTree->refine((size_t)(std::sqrt(std::pow(2, m_iter) * m_sppPerPass / 4) * m_sTreeThreshold), m_sdTreeMaxMemory);
        m_sdTree->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(20, m_dTreeThreshold); });
    }

    void buildSDTree(bool doBuild = true) {
        if (doBuild) {
            Log(EInfo, "Building distributions for sampling.");

            // Build distributions
            m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });
        }

        // Gather statistics
        int maxDepth = 0;
        int minDepth = std::numeric_limits<int>::max();
        Float avgDepth = 0;
        Float maxAvgRadiance = 0;
        Float minAvgRadiance = std::numeric_limits<Float>::max();
        Float avgAvgRadiance = 0;
        size_t maxNodes = 0;
        size_t minNodes = std::numeric_limits<size_t>::max();
        Float avgNodes = 0;
        Float maxStatisticalWeight = 0;
        Float minStatisticalWeight = std::numeric_limits<Float>::max();
        Float avgStatisticalWeight = 0;

        int nPoints = 0;
        int nPointsNodes = 0;

        m_sdTree->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
            const int depth = dTree->depth();
            maxDepth = std::max(maxDepth, depth);
            minDepth = std::min(minDepth, depth);
            avgDepth += depth;

            const Float avgRadiance = dTree->meanRadiance();
            maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
            minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
            avgAvgRadiance += avgRadiance;

            if (dTree->numNodes() > 1) {
                const size_t nodes = dTree->numNodes();
                maxNodes = std::max(maxNodes, nodes);
                minNodes = std::min(minNodes, nodes);
                avgNodes += nodes;
                ++nPointsNodes;
            }

            const Float statisticalWeight = dTree->statisticalWeight();
            maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
            minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
            avgStatisticalWeight += statisticalWeight;

            ++nPoints;
        });

        if (nPoints > 0) {
            avgDepth /= nPoints;
            avgAvgRadiance /= nPoints;

            if (nPointsNodes > 0) {
                avgNodes /= nPointsNodes;
            }

            avgStatisticalWeight /= nPoints;
        }

        Log(EInfo,
            "Distribution statistics:\n"
            "  Depth         = [%d, %f, %d]\n"
            "  Mean radiance = [%f, %f, %f]\n"
            "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
            "  Stat. weight  = [%f, %f, %f]\n",
            minDepth, avgDepth, maxDepth,
            minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
            minNodes, avgNodes, maxNodes,
            minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
        );

        m_isBuilt = true;
    }

    void buildImageStatistics(bool doBuild = true) {
        const Float weight = m_imageStatistics.getWeight();
        const Float avgLR = m_imageStatistics.getAverageLR();
        const Float avgGuiding = m_imageStatistics.getAverageGuiding();
        const Float avgNEE = m_imageStatistics.getAverageNEE();
        if (doBuild) {
            m_imageSqErr = m_imageStatistics.getSquareError();
            m_imageCost = m_imageStatistics.getCost();
            m_imageStatistics.reset();
        }

        Log(EInfo, "MARS statistics:\n"
            "  Actual Cost       = %.3e\n"
            "  Variance per SPP  = %.3e\n"
            "  Cost per SPP      = %.3e\n"
            "  Act. Efficiency   = %.3e\n"
            "  Average LR split  = %.3f\n"
            "  Average Gui split = %.3f\n"
            "  Average NEE split = %.3f\n",
            m_imageCost * weight,
            m_imageSqErr.average(),
            m_imageCost,
            1 / (m_imageCost * m_imageSqErr.average()),
            avgLR,
            avgGuiding,
            avgNEE
        );
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


    void computePixelEstimate(const ref<Film> &film, const Scene *scene) {
        const Vector2i size = film->getSize();

        if (!m_pixelEstimate) {
            m_pixelEstimate = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        }
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
    }

    bool restoreSDTree(Scene* scene) {
        fs::path path = scene->getDestinationFile();
        path = path.parent_path() / (path.leaf().string() + ".sdt");

        if (!m_sdTreePath.empty())
            path = m_sdTreePath;

        BlobReader blob(path.string());
        if (!blob.isValid()) {
            return false;
        }

        /// read pixel estimate
        size_t bufferSize;
        Vector2i pixEstSize;
        blob
            >> pixEstSize.x
            >> pixEstSize.y
            >> bufferSize;
        m_pixelEstimate = new Bitmap(Bitmap::ESpectrumAlphaWeight, Bitmap::EFloat32, pixEstSize, -1);
        blob.Read(m_pixelEstimate->getUInt8Data(), m_pixelEstimate->getBufferSize());

        Matrix4x4 cameraMatrix;

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blob >> (Float&)cameraMatrix(i, j);
            }
        }

        /// read global statistics
        blob >> m_imageCost;
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
            blob >> (Float&)m_imageSqErr[i];

        // read guiding caches
        m_sdTree->read(blob);
        buildSDTree(false);
        buildImageStatistics(false);

        return true;
    }

    void dumpSDTree(Scene* scene, ref<Sensor> sensor) {
        std::ostringstream extension;
        extension << "-" << std::setfill('0') << std::setw(2) << m_iter << ".sdt";
        fs::path path = scene->getDestinationFile();
        path = path.parent_path() / (path.leaf().string() + extension.str());

        if (!m_sdTreePath.empty())
            path = m_sdTreePath;

        auto cameraMatrix = sensor->getWorldTransform()->eval(0).getMatrix();

        BlobWriter blob(path.string());

        /// write out pixel estimate
        blob
            << m_pixelEstimate->getWidth()
            << m_pixelEstimate->getHeight()
            << m_pixelEstimate->getBufferSize();
        blob.Write(m_pixelEstimate->getUInt8Data(), m_pixelEstimate->getBufferSize());

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blob << (Float)cameraMatrix(i, j);
            }
        }

        /// write global statistics
        blob
            << m_imageCost;
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
            blob << (Float)m_imageSqErr[i];

        /// write guiding caches
        m_sdTree->dump(blob);
    }

    bool performRenderPasses(Float& variance, int numPasses, Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID, bool *didAbort = nullptr) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Log(EInfo, "Rendering %d render passes.", numPasses);

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < numPasses; ++i) {
            ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            m_renderProcesses.push_back(process);
        }

        bool result = true;
        int passesRenderedLocal = 0;

        static const size_t processBatchSize = 1;

        for (size_t i = 0; i < m_renderProcesses.size(); i += processBatchSize) {
            const size_t start = i;
            const size_t end = std::min(i + processBatchSize, m_renderProcesses.size());
            for (size_t j = start; j < end; ++j) {
                sched->schedule(m_renderProcesses[j]);
            }

            bool shouldAbort;
            for (size_t j = start; j < end; ++j) {
                auto& process = m_renderProcesses[j];
                sched->wait(process);

                ++m_passesRendered;
                ++m_passesRenderedThisIter;
                ++passesRenderedLocal;

                int progress = 0;
                switch (m_budgetType) {
                    case ESpp:
                        progress = m_passesRendered;
                        shouldAbort = false;
                        break;
                    case ESeconds:
                        progress = (int)computeElapsedSeconds(m_startTime);
                        shouldAbort = progress > m_budget;
                        if (didAbort)
                            *didAbort = shouldAbort;
                        break;
                    default:
                        Assert(false);
                        break;
                }

                m_progress->update(progress);

                if (process->getReturnStatus() != ParallelProcess::ESuccess) {
                    result = false;
                    shouldAbort = true;
                }

            }
            if (shouldAbort) {
                goto l_abort;
            }
        }
    l_abort:

        for (auto& process : m_renderProcesses) {
            sched->cancel(process);
        }

        m_renderProcesses.clear();

        variance = 0;
        Bitmap* squaredImage = m_squaredImage->getBitmap();
        Bitmap* image = m_image->getBitmap();

        if (!m_isFinalIter) {
            computePixelEstimate(film, scene);
        }

        if (m_sampleCombination == ESampleCombination::EInverseVariance) {
            // Record all previously rendered iterations such that later on all iterations can be
            // combined by weighting them by their estimated inverse pixel variance.
            m_images.push_back(image->clone());
        }

        int N = passesRenderedLocal * m_sppPerPass;

        Vector2i size = squaredImage->getSize();
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x) {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);
                Spectrum localVar = squaredImage->getPixel(pos) - pixel * pixel / (Float)N;
                // The local variance is clamped such that fireflies don't cause crazily unstable estimates.
                variance += std::min<Float>(localVar.getLuminance(), 10000.0);
            }

        variance = variance / ((Float)size.x * size.y * (N - 1));

        if (m_sampleCombination == ESampleCombination::EInverseVariance) {
            // Record estimated mean pixel variance for later use in weighting of all images.
            m_variances.push_back(variance);
        }

        Float seconds = computeElapsedSeconds(start);

        const Float ttuv = seconds * variance;
        const Float stuv = passesRenderedLocal * m_sppPerPass * variance;
        Log(EInfo, "%.2f seconds, Total passes: %d, Var: %f, TTUV: %f, STUV: %f.",
            seconds, m_passesRendered, variance, ttuv, stuv);

        return result;
    }

    bool doNeeWithSpp(int spp) {
        switch (m_nee) {
            case ENever:
                return false;
            case EKickstart:
                return spp < 128;
            default:
                return true;
        }
    }

    bool renderSPP(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t sampleCount = (size_t)m_budget;

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        int nPasses = (int)std::ceil(sampleCount / (Float)m_sppPerPass);
        sampleCount = m_sppPerPass * nPasses;

        bool result = true;
        Float currentVarAtEnd = std::numeric_limits<Float>::infinity();

        if (m_dumpSDTree && restoreSDTree(scene)) {
            Log(EInfo, "RESTORED SD TREE");

            Float variance;

            film->clear();
            m_image->clear();
            m_squaredImage->clear();
#ifdef MARS_INCLUDE_AOVS
            m_statsImages->clear();
#endif
            resetPathHistograms();

            m_isFinalIter = true;
            m_doNee = m_nee != ENever;
            m_iter = m_trainingIterations;
            SAssert(m_iter > 3);

            m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", m_budget, job));
            //m_sppPerPass = 1;
            if (!performRenderPasses(variance, m_budget, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                return false;
            }

            return true;
        }

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", nPasses, job));

        while (result && m_passesRendered < nPasses) {
            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            int remainingPasses = nPasses - m_passesRendered;
            int passesThisIteration = std::min(remainingPasses, 1 << m_iter);

            Log(EInfo, "ITERATION %d, %d passes", m_iter, passesThisIteration);

            m_isFinalIter = m_iter >= m_trainingIterations;

            if (m_isFinalIter) {
                passesThisIteration = remainingPasses;
                Log(EInfo, "FINAL %d passes", remainingPasses);
            }

            film->clear();
            m_image->clear();
            m_squaredImage->clear();
#ifdef MARS_INCLUDE_AOVS
            m_statsImages->clear();
#endif
            resetSDTree();
            resetPathHistograms();

            Float variance;
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                result = false;
                break;
            }

            const Float lastVarAtEnd = currentVarAtEnd;
            currentVarAtEnd = passesThisIteration * variance / remainingPasses;

            Log(EInfo,
                "Extrapolated var:\n"
                "  Last:    %f\n"
                "  Current: %f\n",
                lastVarAtEnd, currentVarAtEnd);

            remainingPasses -= passesThisIteration;
            buildSDTree();
            buildImageStatistics();

            if (m_dumpSDTree && !m_isFinalIter) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;
        }

        return result;
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        const Float trainingBudget = m_trainingBudget;
        const Float fullBudget = m_trainingBudget + m_renderingBudget;

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)fullBudget, job));

        const int passesThisIteration = 64;
        m_doNee = true; /// @todo hack

        /// training
        Float variance;
        for (int trainingIteration = 0; trainingIteration < m_trainingIterations; trainingIteration++) {
            film->clear();
            m_image->clear();
            m_squaredImage->clear();
#ifdef MARS_INCLUDE_AOVS
            m_statsImages->clear();
#endif
            resetSDTree();
            resetPathHistograms();

            m_budget = trainingBudget / (1 << (m_trainingIterations - (trainingIteration + 1)));
            Log(EInfo, "ITERATION %d, until %.1f seconds", m_iter, m_budget);

            bool didAbort = false;
            while (!didAbort) {
                if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID, &didAbort)) {
                    return false;
                }
            }

            buildSDTree();
            buildImageStatistics();

            if (m_dumpSDTree && !m_isFinalIter) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;
        }

        film->clear();
        m_image->clear();
        m_squaredImage->clear();
#ifdef MARS_INCLUDE_AOVS
        m_statsImages->clear();
#endif
        resetSDTree();
        resetPathHistograms();

        m_budget = fullBudget;
        m_isFinalIter = true;
        Log(EInfo, "FINAL ITERATION %d, until %.1f seconds", m_iter, m_budget);

        /// rendering
        bool didAbort = false;
        while (!didAbort) {
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID, &didAbort)) {
                return false;
            }
        }

        return true;
    }

    inline bool exists (const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {

        m_sdTree = std::unique_ptr<STree>(new STree(scene->getAABB()));
        m_iter = 0;
        m_isFinalIter = false;

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

#ifdef MARS_INCLUDE_AOVS
        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);

        {
            /// debug film with additional channels
            StatsDescriptor statsDesc;

            auto properties = Properties(film->getProperties());
            properties.setString("pixelFormat", statsDesc.types);
            properties.setString("channelNames", statsDesc.names);
            std::cout << properties.toString() << std::endl;
            auto rfilter = film->getReconstructionFilter();

            m_debugFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
            m_debugFilm->addChild(rfilter);
            m_debugFilm->configure();

            m_statsImages.reset(new StatsImageBlocks([&]() {
                return new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
            }));
            m_debugImage = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, film->getCropSize(), NULL,
                statsDesc.size * SPECTRUM_SAMPLES + 2
            );
        }
#endif

        renderDenoisingAuxiliaries(scene, queue, job, sceneResID, sensorResID);

        m_squaredImage = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
        m_image = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());

        m_images.clear();
        m_variances.clear();

        /* Just dump some info to have it available in the exr file */
        Log(EInfo, "Configuration:\n"\
                   "  MARS latest commit hash: %s\n"
                   "  RenderMethod: %s\n"
                   "  splitConfig:  %s\n"
                   "  Max depth:    %d\n"
                   "  RR depth:     %d\n"
                   "  splitMin:     %.2f\n"
                   "  splitMax:     %.2f\n"
                   "  tBudget:      %.2f\n"
                   "  rBudget:      %.2f\n"
                   "  Combination:  %s\n"
                   "  Learning:     %s\n"
                   "  NEE:          %s\n"
                   "  Precision:    %s\n"
                   "  branchMax:    %.2f\n"
                   "  AOVs:         %s\n"
                   "  Rounding:     %s",
                g_GIT_SHA1,
                errsToTxt[(int)m_rrsMode],
                m_rrsMode == EMARS ? splitConfigToText[(int)m_splitConfig] : "none",
                m_maxDepth,
                m_rrDepth,
                m_splittingMin,
                m_splittingMax,
                m_trainingBudget,
                m_renderingBudget,
                m_sampleCombinationStr.c_str(),
                m_bsdfSamplingFractionLossStr.c_str(),
                m_neeStr.c_str(),
#ifdef SINGLE_PRECISION
                "SINGLE",
#else // DOUBLE_PRECISION
                "DOUBLE",
#endif
                m_branchingMax,
#ifdef MARS_INCLUDE_AOVS
                "ENABLED",
#else
                "DISABLED",
#endif
#ifdef STOCHASTIC_DIV_NUM_SAMPLES
                "STOCHASTIC (DIV ROUNDED)"
#elif defined(LOW_DISCREPANCY_NUM_SAMPLES)
                "LOW DISCREPANCY"
#else
                "STOCHASTIC (DIV STOCHASTIC)"
#endif
                );

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        /* Check if output path exists, otherwise create the requested directory */
        if (!exists(scene->getDestinationFile().parent_path().string())) {
            if (mkdir(scene->getDestinationFile().parent_path().string().c_str(), 0775) == -1)
                exit(1);
        }

        m_startTime = std::chrono::steady_clock::now();

        m_passesRendered = 0;
        switch (m_budgetType) {
            case ESpp:
                result = renderSPP(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                break;
            case ESeconds:
                result = renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                break;
            default:
                Assert(false);
                break;
        }

        sched->unregisterResource(integratorResID);

        m_progress = nullptr;

        if (m_sampleCombination == ESampleCombination::EInverseVariance) {
            // Combine the last 4 images according to their inverse variance
            film->clear();
            // SAssert(!"not tested");

            ref<ImageBlock> tmp = new ImageBlock(Bitmap::ESpectrum, film->getCropSize());
            size_t begin = m_images.size() - std::min(m_images.size(), (size_t)4);

            Float totalWeight = 0;
            for (size_t i = begin; i < m_variances.size(); ++i) {
                totalWeight += 1.0f / m_variances[i];
            }

            for (size_t i = begin; i < m_images.size(); ++i) {
                m_images[i]->convert(tmp->getBitmap(), 1.0f / m_variances[i] / totalWeight);
                film->addBitmap(tmp->getBitmap());
            }
        }

        {
            /// output path length statistics
            std::stringstream ss;
            for (int i = 0; i < int(m_pathLengthHistogram.size()); ++i) {
                if (i) ss << ", ";
                ss << m_pathLengthHistogram[i];
            }

            SLog(EInfo, "Path length histogram: [%s]", ss.str().c_str());
        }

        {
            /// output contribution statistics
            std::stringstream ss;
            for (int i = 0; i < int(m_contributionHistogram.size()); ++i) {
                if (i) ss << ", ";
                ss << m_contributionHistogram[i];
            }

            SLog(EInfo, "Contribution histogram: [%s]", ss.str().c_str());
        }

#ifdef MARS_INCLUDE_AOVS
        Bitmap* image = m_image->getBitmap();
        auto statsBitmaps = m_statsImages->getBitmaps();
        Float* debugImage = m_debugImage->getBitmap()->getFloatData();

        Vector2i size = image->getSize();
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x) {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);

                /// write out debug channels
                for (int i = 0; i < SPECTRUM_SAMPLES; ++i) *(debugImage++) = pixel[i];

                int sbIndex = 0;
                for (auto &b : statsBitmaps) {
                    Spectrum v = b->getPixel(pos);
                    if (sbIndex++ == 2)
                        /// normalize relvar by diving through spp.
                        v /= m_passesRenderedThisIter;
                    for (int i = 0; i < SPECTRUM_SAMPLES; ++i) *(debugImage++) = v[i];
                }

                *(debugImage++) = 1.0f;
                *(debugImage++) = 1.0f;
            }

        m_debugFilm->setBitmap(m_debugImage->getBitmap());

        {
            /// output debug image
            std::string suffix = "-dbg-" + m_rrsStr;
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

        return result;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
        Sampler *sampler, ImageBlock *block, const bool &stop,
        const std::vector< TPoint2<uint8_t> > &points) const {

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float)m_sppPerPass);

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

#ifdef MARS_INCLUDE_AOVS
        static thread_local StatsImageBlocks blocks([&]() {
            auto b = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
            return b;
        });

        for (auto &b : blocks.blocks) {
            b->setOffset(block->getOffset());
            b->clear();
        }
#endif

        EstimatorStatistics<double> estimatorStatistics;

        ref<ImageBlock> squaredBlock = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
        squaredBlock->setOffset(block->getOffset());
        squaredBlock->clear();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        std::vector<double> localContributionHistogram;
        localContributionHistogram.resize(m_maxDepth);
        std::fill(localContributionHistogram.begin(), localContributionHistogram.end(), 0);

        StatsValues stats;
        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;

            Spectrum pixelEstimate { 0.5f };
            if (m_pixelEstimate.get()) {
                pixelEstimate = m_pixelEstimate->getPixel(offset);
            }

            stats.pixelEstimate.value = pixelEstimate;
            const Spectrum metricNorm = m_useAbsoluteMetric ? Spectrum { 1.f } : pixelEstimate + Spectrum { 1e-2 };
            const Spectrum expectedContribution = pixelEstimate / metricNorm;//.average();

            for (int j = 0; j < m_sppPerPass; j++) {
                stats.reset();

                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                LiInput input;
                input.liEstimate = pixelEstimate;
                input.throughput = spec;
                input.relativeThroughput = spec / metricNorm;//.average();
                input.ray = sensorRay;
                input.rRec = rRec;

                LiOutput output = Li(input, stats, localContributionHistogram);
                output.cost += COST_PRIMARY;
                spec *= output.totalContribution();
                block->put(samplePos, spec, rRec.alpha);
                squaredBlock->put(samplePos, spec * spec, rRec.alpha);
                sampler->advance();

                Spectrum contribution = output.totalContribution() * input.relativeThroughput;
                Float outlierHeuristic = contribution.average();
                outlierHeuristic = OUTLIER_MAX == -1 ?
                        1 : std::max(Float(1), outlierHeuristic / OUTLIER_MAX);
                estimatorStatistics.addSample(1, contribution / outlierHeuristic, output.cost);

                for (int chan = 0; chan < SPECTRUM_SAMPLES; ++chan)
                    contribution[chan] = contribution[chan] / outlierHeuristic - expectedContribution[chan];

#ifdef MARS_INCLUDE_AOVS
                stats.relvar.value = contribution * contribution;
                stats.pixelCost.value = output.cost * 1e+6;
                stats.avgPathLength.add(output.averagePathLength()-1);
                stats.put(blocks, samplePos, rRec.alpha);
#endif
                estimatorStatistics.addSplitStatistics(output.lrSamples, output.guidingSamples, output.neeSamples);
            }
        }

        if (!stop) {
            m_squaredImage->put(squaredBlock);
            m_image->put(block);
#ifdef MARS_INCLUDE_AOVS
            m_statsImages->put(blocks);
#endif

            m_imageStatistics += estimatorStatistics;

            for (int i = 0; i < m_maxDepth; ++i) {
                atomicAdd(&m_contributionHistogram[i], localContributionHistogram[i]);
            }
        }
    }

    void cancel() {
        const auto& scheduler = Scheduler::getInstance();
        for (size_t i = 0; i < m_renderProcesses.size(); ++i) {
            scheduler->cancel(m_renderProcesses[i]);
        }
    }

    Spectrum sampleMat(const BSDF* bsdf, BSDFSamplingRecord& bRec, Float& woPdf, Float& bsdfPdf, Float& dTreePdf, Float bsdfSamplingFraction, RadianceQueryRecord& rRec, const DTreeWrapper* dTree, bool& guided) const {
        Point2 sample = rRec.nextSample2D();

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            auto result = bsdf->sample(bRec, bsdfPdf, sample);
            woPdf = bsdfPdf;
            dTreePdf = 0;
            return result;
        }

        /* We need to split this case because we want dTreePdf to be set for splitting BSDF/Guiding */
        if (bsdfSamplingFraction == 1) {
            auto result = bsdf->sample(bRec, bsdfPdf, sample);
            woPdf = bsdfPdf;
            /* Fill dTreePdf for splitting BSDF/Guiding */
            const bool sampledDelta = bRec.sampledType & BSDF::EDelta;
            dTreePdf = !sampledDelta ? dTree->pdf(bRec.its.toWorld(bRec.wo)) : 0;
            return result;
        }

        Spectrum result;
        if (sample.x < bsdfSamplingFraction) {
            sample.x /= bsdfSamplingFraction;
            result = bsdf->sample(bRec, bsdfPdf, sample);
            if (result.isZero()) {
                woPdf = bsdfPdf = dTreePdf = 0;
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta) {
                dTreePdf = 0;
                woPdf = bsdfPdf * bsdfSamplingFraction;
                return result / bsdfSamplingFraction;
            }

            result *= bsdfPdf;
        } else {
            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);
            guided = true;
            bRec.wo = bRec.its.toLocal(dTree->sample(rRec.sampler));
            result = bsdf->eval(bRec);
            bRec.eta = Frame::cosTheta(bRec.wi) > 0.0f ? bsdf->getEta() : 1.0f / bsdf->getEta();
        }

        pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, dTree);
        if (woPdf == 0) {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    void pdfMat(Float& woPdf, Float& bsdfPdf, Float& dTreePdf, Float bsdfSamplingFraction, const BSDF* bsdf, const BSDFSamplingRecord& bRec, const DTreeWrapper* dTree) const {
        dTreePdf = 0;

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            return;
        }

        /* We need to split this case because we want dTreePdf to be set for splitting BSDF/Guiding */
        if (bsdfSamplingFraction == 1) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            const bool sampledDelta = bRec.sampledType & BSDF::EDelta;
            dTreePdf = !sampledDelta ? dTree->pdf(bRec.its.toWorld(bRec.wo)) : 0;
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        if (!std::isfinite(bsdfPdf)) {
            woPdf = 0;
            return;
        }

        dTreePdf = dTree->pdf(bRec.its.toWorld(bRec.wo));
        woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * dTreePdf;
    }

    Float clamp(Float sf) const {
        sf = std::min(sf, m_splittingMax);
        sf = std::max(sf, m_splittingMin);
        return sf;
    }

    Spectrum Li(const RayDifferential &, RadianceQueryRecord &) const {
        SAssert(!"Li() cannot be used directly in this integrator!");
        return Spectrum { 0.f };
    }

    inline void roundSF(RadianceQueryRecord& rRec, Float& sfBSDF, Float& sfNEE, Float& sfGuiding,
                 int& numSamplesBSDF, int& numSamplesNEE, int& numSamplesGuiding) const {
#ifdef STOCHASTIC_DIV_NUM_SAMPLES
        /// We need to change the splitting factors in this case as well to stay unbiased
        Float tmp = rRec.nextSample1D();
        numSamplesBSDF = std::floor(tmp += sfBSDF);
        if (sfBSDF > 1)
            sfBSDF = numSamplesBSDF;

        numSamplesGuiding = std::floor(tmp += sfGuiding - numSamplesBSDF);
        if (sfGuiding > 1)
            sfGuiding = numSamplesGuiding;

        numSamplesNEE = std::floor(tmp += sfNEE - numSamplesGuiding);
        if (sfNEE > 1)
            sfNEE = numSamplesNEE;

#elif defined(LOW_DISCREPANCY_NUM_SAMPLES)
        Float tmp = rRec.nextSample1D();
        numSamplesBSDF = std::floor(tmp += sfBSDF);
        numSamplesGuiding = std::floor(tmp += sfGuiding - numSamplesBSDF);
        numSamplesNEE = std::floor(tmp += sfNEE - numSamplesGuiding);
#else
        numSamplesBSDF = std::floor(sfBSDF + rRec.nextSample1D());
        numSamplesGuiding = std::floor(sfGuiding + rRec.nextSample1D());
        numSamplesNEE = std::floor(sfNEE + rRec.nextSample1D());
#endif
    }

    LiOutput Li(LiInput &input, StatsValues &stats, std::vector<double> &contributionHistogram) const {
        LiOutput output;

        struct Vertex
        {
            DTreeWrapper* dTree;
            Vector dTreeVoxelSize;
            Ray ray;

            Spectrum throughput;
            Spectrum relativeThroughput; // not including last bsdf weight
            Spectrum bsdfVal;
            Float cosTheta;

            Spectrum radiance;

            Float woPdf, bsdfPdf, dTreePdf;
            bool isDelta;

            void record(const Spectrum& r) {
                radiance += r;
            }

            void commit(STree& sdTree, Float statisticalWeight, ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler* sampler) {
                if (!(woPdf > 0) || !radiance.isValid() || !bsdfVal.isValid()) {
                    return;
                }

                Spectrum localRadiance = Spectrum{0.0f};
                if (throughput[0] * woPdf > Epsilon) localRadiance[0] = radiance[0] / throughput[0];
                if (throughput[1] * woPdf > Epsilon) localRadiance[1] = radiance[1] / throughput[1];
                if (throughput[2] * woPdf > Epsilon) localRadiance[2] = radiance[2] / throughput[2];
                Spectrum product = localRadiance * bsdfVal;

                DTreeRecord rec {
                    ray.d,
                    localRadiance.average() * cosTheta, // radiance
                    product.average(), // product
                    (radiance * relativeThroughput * woPdf).average(), // image contribution
                    woPdf, bsdfPdf, dTreePdf,
                    statisticalWeight, isDelta
                };
                switch (spatialFilter) {
                    case ESpatialFilter::ENearest:
                        dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        break;
                    case ESpatialFilter::EStochasticBox:
                        {
                            DTreeWrapper* splatDTree = dTree;

                            // Jitter the actual position within the
                            // filter box to perform stochastic filtering.
                            Vector offset = dTreeVoxelSize;
                            offset.x *= sampler->next1D() - 0.5f;
                            offset.y *= sampler->next1D() - 0.5f;
                            offset.z *= sampler->next1D() - 0.5f;

                            Point origin = sdTree.aabb().clip(ray.o + offset);
                            splatDTree = sdTree.dTreeWrapper(origin);
                            if (splatDTree) {
                                splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                            }
                            break;
                        }
                    case ESpatialFilter::EBox:
                        sdTree.record(ray.o, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
                        break;
                }
            }
        };

        RadianceQueryRecord &rRec = input.rRec;
        if (m_maxDepth >= 0 && input.rRec.depth > m_maxDepth) {
            // maximum depth reached
terminatePath:
            /// please excuse this hack...
            //uint64_t &l = m_pathLengthHistogram[input.rRec.depth-1];
            //reinterpret_cast<std::atomic<uint64_t> &>(l)++;

            output.markAsLeaf(rRec.depth);

            return output;
        }

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray { input.ray };
        Spectrum transmittance { 1.f };

        if (!input.scattered) {
            /* Perform the first ray intersection (or ignore if the
            intersection has already been provided). */
            rRec.rayIntersect(ray);
        }

        /* ==================================================================== */
        /*                 Radiative Transfer Equation sampling                 */
        /* ==================================================================== */
        if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler))
        {
            SAssert(!"medium support has not been tested");
            /* Sample the integral
            \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
            */
            const PhaseFunction *phase = mRec.getPhaseFunction();

            if (rRec.depth >= m_maxDepth && m_maxDepth != -1) // No more scattering events allowed
                goto terminatePath;

            transmittance *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

            /* ==================================================================== */
            /*                          Luminaire sampling                          */
            /* ==================================================================== */

            /* Estimate the single scattering component if this is requested */
            DirectSamplingRecord dRec(mRec.p, mRec.time);

            if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance) {
                int interactions = m_maxDepth - rRec.depth - 1;

                Spectrum value = scene->sampleAttenuatedEmitterDirect(
                    dRec, rRec.medium, interactions,
                    rRec.nextSample2D(), rRec.sampler);

                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Evaluate the phase function */
                    PhaseFunctionSamplingRecord pRec(mRec, -ray.d, dRec.d);
                    Float phaseVal = phase->eval(pRec);

                    if (phaseVal != 0) {
                        /* Calculate prob. of having sampled that direction using
                        phase function sampling */
                        Float phasePdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? phase->pdf(pRec) : (Float) 0.0f;

                        /* Weight using the power heuristic */
                        const Float weight = miWeight(dRec.pdf, phasePdf);
                        output.reflected += transmittance * value * phaseVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /*                         Phase function sampling                      */
            /* ==================================================================== */

            Float phasePdf;
            PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
            Float phaseVal = phase->sample(pRec, phasePdf, rRec.sampler);
            if (phaseVal == 0)
                goto terminatePath;
            transmittance *= phaseVal;

            /* Trace a ray in this direction */
            ray = Ray(mRec.p, pRec.wo, ray.time);
            ray.mint = 0;

            Spectrum value(0.0f);
            rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

            /* If a luminaire was hit, estimate the local illumination and
            weight using the power heuristic */
            if (!value.isZero() && (rRec.type & RadianceQueryRecord::EDirectMediumRadiance)) {
                const Float emitterPdf = scene->pdfEmitterDirect(dRec);
                output.reflected += transmittance * value * miWeight(phasePdf, emitterPdf);
            }

            /* ==================================================================== */
            /*                         Multiple scattering                          */
            /* ==================================================================== */

            /* Stop if multiple scattering was not requested */
            if (!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
                goto terminatePath;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                while accounting for the solid angle compression at refractive
                index boundaries. Stop with at least some probability to avoid
                getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(input.relativeThroughput.max() * input.eta * input.eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    goto terminatePath;

                transmittance /= q;
            }

            SAssert(!"not yet implemented");
        }
        else
        {
            /* Sample
            tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
            Account for this and multiply by the proper per-color-channel transmittance.
            */
            if (rRec.medium) {
                transmittance *= mRec.transmittance / mRec.pdfFailure;
            }

            if (!its.isValid()) {
                /* If no intersection could be found, possibly return
                attenuated radiance from a background luminaire */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || input.scattered)) {
                    Spectrum value = transmittance * scene->evalEnvironment(ray);
                    if (rRec.medium)
                        value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
                    output.emitted += value;

                    atomicAdd(&m_pathLengthHistogram[rRec.depth-1], 1);
                    contributionHistogram[rRec.depth-1] += (value * input.relativeThroughput).average();
                }

                goto terminatePath;
            }

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || input.scattered)) {
                Spectrum value = transmittance * its.Le(-ray.d);
                output.emitted += value;

                atomicAdd(&m_pathLengthHistogram[rRec.depth-1], 1);
                contributionHistogram[rRec.depth-1] += (value * input.relativeThroughput).average();
            }

            /* Include radiance from a subsurface integrator if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                Spectrum value = transmittance * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);
                output.emitted += value;

                atomicAdd(&m_pathLengthHistogram[rRec.depth-1], 1);
                contributionHistogram[rRec.depth-1] += (value * input.relativeThroughput).average();
            }

            if (m_maxDepth >= 0 && input.rRec.depth >= m_maxDepth)
                goto terminatePath;

            /* Prevent light leaks due to the use of shading normals */
            Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                wiDotShN = Frame::cosTheta(its.wi);
            if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
                goto terminatePath;

            const BSDF *bsdf = its.getBSDF();
            const Spectrum albedo = bsdf->getDiffuseReflectance(its) + bsdf->getSpecularReflectance(its);
            const bool hasTransmission = bsdf->getType() & BSDF::ETransmission ? true : false;
            const bool bsdfHasSmoothComponent = bsdf->getType() & BSDF::ESmooth;

            Vector dTreeVoxelSize;
            DTreeWrapper* dTree = nullptr;
            int binIndex = 0;
            Spectrum LiEstimate { 0.f };
            Spectrum LocalLiEstimate { 0.f };

            /// collect auxilliary information for debugging and denoising
            if (bsdfHasSmoothComponent) {
                Spectrum normal;

                Vector n = its.shFrame.n;
                for (int i = 0; i < 3; ++i) normal[i] = n[i];

                stats.albedo.add(rRec.depth-1, input.throughput * albedo);
                stats.normal.add(rRec.depth-1, input.throughput * normal);
            }

            ERRS rrsMode = m_rrsMode;
            if (m_iter < 3)
                // don't use learning based methods unless caches have somewhat converged
                rrsMode = EClassicRRS;// ENoRRS;

            Float roughness = std::numeric_limits<Float>::infinity();
            for (int comp = 0; comp < bsdf->getComponentCount(); ++comp) {
                roughness = std::min(roughness, bsdf->getRoughness(its, comp));
            }

            // We only guide smooth BRDFs for now. Analytic product sampling
            // would be conceivable for discrete decisions such as refraction vs
            // reflection.
            if (bsdfHasSmoothComponent) {
                dTree = m_sdTree->dTreeWrapper(its.p, dTreeVoxelSize);
                binIndex = dTree->mapContextToIndex(
                    Float(wiDotShN > 0 ? 1 : -1) * its.shFrame.n,
                    input.ray.d,
                    roughness,
                    m_rrsMode == EMARS
                );

                // use conservative estimate for outgoing radiance
                if (rrsMode == EADRRS2) {
                    LocalLiEstimate = dTree->getIrradianceEstimate(binIndex);
                    LiEstimate = LocalLiEstimate;
                } else {
                    LocalLiEstimate = albedo * dTree->getIrradianceEstimate(binIndex) / M_PI;
                    LiEstimate = input.liEstimate.average() > LocalLiEstimate.average() ?
                        input.liEstimate :
                        LocalLiEstimate
                    ;
                }
            }

            Float splittingFactorBSDF = 1;
            Float splittingFactorNEE = 1;
            Float splittingFactorGuiding = 0;
            Float weightWindow = 1;

            switch (rrsMode) {
            case ENoRRS:
                break;

            case EAlbedoRR:
                if (rRec.depth >= m_rrDepth) {
                    splittingFactorBSDF = input.relativeThroughput.max() * input.eta * input.eta * input.bsdfPdfCorrection;
                }
                break;

            case EAlbedo2RRS:
                if (rRec.depth >= m_rrDepth && !albedo.isZero()) {
                    splittingFactorBSDF = albedo.average();
                }
                break;

            case EClassicRRS:
                if ((rRec.depth >= CLASSIC_RR_TRAIN_DEPTH && m_rrsMode != EClassicRRS) ||
                    (rRec.depth >= m_rrDepth)) {
                    splittingFactorBSDF = std::min<Float>(0.95, input.relativeThroughput.max() * input.eta * input.eta);
                }
                break;

            case EADRRS:
            case EADRRS2:
                if (input.scattered && LiEstimate.max() > 0 && !hasTransmission) {
                    // do not perform ADRRS on transmissive surfaces:
                    // on those ADRRS prefers russian roulette and causes excessive variance
                    // (see Robust Fitting of Parallax-Aware Mixtures for Path Guiding)
                    splittingFactorBSDF = (input.relativeThroughput * LiEstimate).average();
                    splittingFactorNEE = splittingFactorBSDF;
                    weightWindow = 1;
                }
                break;

            case EMARS:
                if (
                    dTree &&
                    dTree->hasPrimaryEstimates(binIndex)
                ) {

                    const Spectrum& tp = input.relativeThroughput;
                    const Float img_marsFactor = std::sqrt(m_imageCost / m_imageSqErr.average());

                    auto getCombinedSF = [&](const Spectrum& efR, const Spectrum& efS) {
                        const Float sfR = transmittance.average() *
                                                            std::sqrt( (tp * tp * efR).average() ) * img_marsFactor;
                        const Float sfS = transmittance.average() *
                                                            std::sqrt( (tp * tp * efS).average() ) * img_marsFactor;
                        if (sfR > 1) {
                            if (sfS < 1) {
                                /// second moment and variance disagree on whether to split or RR, resort to doing nothing.
                                return clamp(1);
                            } else {
                                /// use variance only if both modes recommend splitting.
                                return clamp(sfS);
                            }
                        } else {
                            /// use second moment only if it recommends RR.
                            return clamp(sfR);
                        }
                    };

                    switch(m_splitConfig) {
                        case BsdfNeeShared: {
                            const Float cost = dTree->getPrimaryCost(binIndex);
                            const Spectrum marsFactor_R = cost > 0 ?
                                                    dTree->getPrimary2ndMom(binIndex) / cost :
                                                    Spectrum{ 0.0 };
                            const Spectrum marsFactor_S = cost > 0 ?
                                                    dTree->getPrimaryVariance(binIndex) / cost :
                                                    Spectrum{ 0.0 };

                            splittingFactorBSDF = splittingFactorNEE = getCombinedSF(marsFactor_R, marsFactor_S);
                            break;
                        }

                        case BsdfNeeGuidingOneSampleSplit: {
                            { /* BSDF + GUIDING */
                                const Float fBsdf = std::sqrt((tp * tp * dTree->getPrimary2ndMomBsdf(binIndex)).average());
                                const Float fGuiding = std::sqrt((tp * tp * dTree->getPrimary2ndMomGuiding(binIndex)).average());
                                /// This assumes constant costs!
                                const Float bsdfSamplingFraction = math::clamp(fBsdf / (fBsdf + fGuiding), Float(0.05), Float(0.95));

                                // compute hypothetical combined estimator statistics
                                const Spectrum primary2ndMom =
                                    dTree->getPrimary2ndMomBsdf(binIndex) / bsdfSamplingFraction +
                                    dTree->getPrimary2ndMomGuiding(binIndex) / (1 - bsdfSamplingFraction);
                                const Spectrum primaryVariance = primary2ndMom -
                                    dTree->getPrimary1stMomBG(binIndex) * dTree->getPrimary1stMomBG(binIndex);
                                const Float primaryCost =
                                    bsdfSamplingFraction * dTree->getPrimaryCostBsdf(binIndex) +
                                    (1 - bsdfSamplingFraction) * dTree->getPrimaryCostGuiding(binIndex);

                                const Spectrum marsFactor_R = primaryCost > 0 ?
                                                        primary2ndMom / primaryCost:
                                                        Spectrum{ 0.0 };
                                const Spectrum marsFactor_S = primaryCost > 0 ?
                                                        primaryVariance / primaryCost :
                                                        Spectrum{ 0.0 };

                                splittingFactorBSDF = splittingFactorGuiding = getCombinedSF(marsFactor_R, marsFactor_S);
                                splittingFactorBSDF *= bsdfSamplingFraction;
                                splittingFactorGuiding *= 1 - bsdfSamplingFraction;
                            }
                            { /* NEE */
                                const Float cost = dTree->getPrimaryCostNee(binIndex);
                                const Spectrum neeFactor_R = cost > 0 ?
                                                        dTree->getPrimary2ndMomNee(binIndex) / cost :
                                                        Spectrum { 0.0 };
                                const Spectrum neeFactor_S = cost > 0 ?
                                                        dTree->getPrimaryVarianceNee(binIndex) / cost :
                                                        Spectrum { 0.0 };

                                splittingFactorNEE = getCombinedSF(neeFactor_R, neeFactor_S);
                            }
                            break;
                        }

                        case BsdfNeeGuidingSplit:
                        case BsdfNeeGuidingSplitSF: {
                            { /* GUIDING */
                                const Float cost = dTree->getPrimaryCostGuiding(binIndex);
                                const Spectrum guidingFactor_R = cost > 0 ?
                                                        dTree->getPrimary2ndMomGuiding(binIndex) / cost :
                                                        Spectrum{ 0.0 };
                                const Spectrum guidingFactor_S = cost > 0 ?
                                                        dTree->getPrimaryVarianceGuiding(binIndex) / cost :
                                                        Spectrum{ 0.0 };

                                splittingFactorGuiding = getCombinedSF(guidingFactor_R, guidingFactor_S);
                            }
                        }
                        /// Fallthrough
                        case BsdfNeeSplit:
                        case BsdfNeeSplitSF: {
                            { /* BSDF */
                                const Float cost = dTree->getPrimaryCostBsdf(binIndex);
                                const Spectrum bsdfFactor_R = cost > 0 ?
                                                        dTree->getPrimary2ndMomBsdf(binIndex) / cost:
                                                        Spectrum{ 0.0 };
                                const Spectrum bsdfFactor_S = cost > 0 ?
                                                        dTree->getPrimaryVarianceBsdf(binIndex) / cost :
                                                        Spectrum{ 0.0 };

                                splittingFactorBSDF = getCombinedSF(bsdfFactor_R, bsdfFactor_S);
                            }
                            { /* NEE */
                                const Float cost = dTree->getPrimaryCostNee(binIndex);
                                const Spectrum neeFactor_R = cost > 0 ?
                                                        dTree->getPrimary2ndMomNee(binIndex) / cost :
                                                        Spectrum { 0.0 };
                                const Spectrum neeFactor_S = cost > 0 ?
                                                        dTree->getPrimaryVarianceNee(binIndex) / cost :
                                                        Spectrum { 0.0 };

                                splittingFactorNEE = getCombinedSF(neeFactor_R, neeFactor_S);
                            }
                            break;
                        }
                        default: Assert(false);
                    }

                    weightWindow = 1;
                }
                break;
            }

            /* Make sure guiding splits are deactivated if not requested */
            const bool splitGuiding =
                m_splitConfig == BsdfNeeGuidingSplit ||
                m_splitConfig == BsdfNeeGuidingSplitSF ||
                m_splitConfig == BsdfNeeGuidingOneSampleSplit;
            const bool budgetAware =
                m_splitConfig == BsdfNeeGuidingSplitSF ||
                m_splitConfig == BsdfNeeSplitSF ||
                m_splitConfig == BsdfNeeGuidingOneSampleSplit;
            Assert(splitGuiding || splittingFactorGuiding == 0);

            for (int sfi = 0; sfi < 2; ++sfi) {
                Float& splittingFactor = sfi == 0 ? splittingFactorBSDF : splittingFactorNEE;
                if (weightWindow > 1) {
                    const Float dminus = 2 / (1 + weightWindow);
                    const Float dplus = weightWindow * dminus;

                    if (splittingFactor < dminus)
                        // russian roulette
                        splittingFactor /= dminus;
                    else if (splittingFactor > dplus)
                        // splitting
                        splittingFactor /= dplus;
                    else
                        // within weight window
                        splittingFactor = 1;
                }

                if (!bsdfHasSmoothComponent)
                    /// @todo hack! ideally we would like to not do RRS on delta components and do smooth components separately!
                    splittingFactor = std::max<Float>(splittingFactor, 1.f);

                splittingFactor = clamp(splittingFactor);
            }
            const Float branchCorrection = std::isinf(m_branchingMax) ? 1.0 :
                std::min(Float(1), m_branchingMax / (input.breadth * (splittingFactorBSDF + splittingFactorGuiding)));
            splittingFactorBSDF *= branchCorrection;
            splittingFactorGuiding *= branchCorrection;

            if (!bsdfHasSmoothComponent) {
                splittingFactorNEE = 0.0;
            }

            if (splitGuiding && rrsMode != EClassicRRS) {
                if (bsdfHasSmoothComponent) {
                    splittingFactorGuiding = clamp(splittingFactorGuiding);
                } else {
                    /* No guiding on delta components afaik. */
                    splittingFactorGuiding = 0.0;
                }
            }

            /* --- BSDF SAMPLING FRACTION --- */
            Float bsdfSamplingFraction = m_bsdfSamplingFraction;
            if (m_disableGuiding) {
                splittingFactorGuiding = 0;
                bsdfSamplingFraction = 1;
            }

            int numSamplesBSDF, numSamplesGuiding, numSamplesNEE;
            roundSF(rRec, splittingFactorBSDF, splittingFactorNEE, splittingFactorGuiding,
                    numSamplesBSDF, numSamplesNEE, numSamplesGuiding);
            if (rRec.depth == 1){
                output.lrSamples = numSamplesBSDF;
                output.guidingSamples = numSamplesGuiding;
                output.neeSamples = numSamplesNEE;
                // Log(EInfo, "rrsMode: %f", splittingFactorBSDF);
            }

            if (splitGuiding) {
                /* Adjust sampling fraction for learning purposes */
                bsdfSamplingFraction = splittingFactorBSDF / (splittingFactorBSDF + splittingFactorGuiding);
            } else if (dTree && m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) {
                bsdfSamplingFraction = dTree->bsdfSamplingFraction();
            }

            if (dTree) {
                stats.primaryCost.add(rRec.depth-1, dTree->getPrimaryCost(binIndex) * 1e+6);
                stats.primaryCostBsdf.add(rRec.depth-1, dTree->getPrimaryCostBsdf(binIndex) * 1e+6);
                stats.primaryCostGuid.add(rRec.depth-1, dTree->getPrimaryCostGuiding(binIndex) * 1e+6);
                stats.primaryCostNee.add(rRec.depth-1, dTree->getPrimaryCostNee(binIndex) * 1e+6);
                stats.primary2ndMom.add(rRec.depth-1, dTree->getPrimary2ndMom(binIndex));
                stats.primary2ndMomBsdf.add(rRec.depth-1, dTree->getPrimary2ndMomBsdf(binIndex));
                stats.primary2ndMomGuid.add(rRec.depth-1, dTree->getPrimary2ndMomGuiding(binIndex));
                stats.primary2ndMomNee.add(rRec.depth-1, dTree->getPrimary2ndMomNee(binIndex));
                stats.primaryVar.add(rRec.depth-1, dTree->getPrimaryVariance(binIndex));
                stats.primaryVarBsdf.add(rRec.depth-1, dTree->getPrimaryVarianceBsdf(binIndex));
                stats.primaryVarGuid.add(rRec.depth-1, dTree->getPrimaryVarianceGuiding(binIndex));
                stats.primaryVarNee.add(rRec.depth-1, dTree->getPrimaryVarianceNee(binIndex));
                // stats.irradiance.add(rRec.depth-1, input.throughput * LocalLiEstimate);
                // stats.irradiance.add(rRec.depth-1, input.throughput * dTree->getIrradianceEstimate(binIndex));
                stats.irradiance.add(rRec.depth-1, input.throughput * albedo * dTree->getIrradianceEstimate(binIndex) / M_PI);
                stats.indirectIrradiance.add(rRec.depth-1, input.throughput * albedo * dTree->getIndirectIrradianceEstimate() / M_PI);
                stats.irradianceWeight.add(rRec.depth-1, dTree->getIrradianceWeight(binIndex) * 1e-6);
                stats.bsdfFraction.add(rRec.depth-1, bsdfSamplingFraction);
            }

            stats.liEstimate.add(rRec.depth-1, input.throughput * input.liEstimate);
            stats.roughness.add(rRec.depth-1, roughness);

            if (splitGuiding || !bsdfHasSmoothComponent) {
                stats.splittingFactorB.add(rRec.depth-1, splittingFactorBSDF);
                stats.splittingFactorG.add(rRec.depth-1, splittingFactorGuiding);
            } else {
                stats.splittingFactorB.add(rRec.depth-1, splittingFactorBSDF * bsdfSamplingFraction);
                stats.splittingFactorG.add(rRec.depth-1, splittingFactorBSDF * (1-bsdfSamplingFraction));
            }
            stats.splittingFactorN.add(rRec.depth-1, splittingFactorNEE);


            if (m_addAdjointEstimate && m_iter >= 3)
                Assert(false);
                //output.reflected += transmittance * (splittingFactor - numSamples) * LocalLiEstimate;
            //if (numSamples == 0) {
            //    output.reflected /= splittingFactor;
            //    goto terminatePath;
            //}

            Spectrum irradianceAccumulator { 0.f };
            Spectrum indirectIrradianceAccumulator { 0.f };

            /* Set MIS multiplier to include splitting factors if requested.
               For guiding, the multiplier should be 0 in case of delta components and if guiding split is disabled.
               Thus, we need the extra check of '== 0' which handles both cases simultaneously. */
            const Float misMultNee  = budgetAware ? splittingFactorNEE  : 1.0;
            const Float misMultBsdf = budgetAware ? splittingFactorBSDF : 1.0;
            const Float misMultGuid = splittingFactorGuiding != 0 ? (budgetAware ? splittingFactorGuiding : 1.0)
                                                                  : 0.0;

            for (int sampleIndex = 0; sampleIndex < numSamplesNEE; ++sampleIndex) {
                LiInput input2 = input;
                input2.breadth *= numSamplesBSDF + numSamplesGuiding;
                input2.throughput *= transmittance / splittingFactorNEE;
                input2.relativeThroughput *= transmittance / splittingFactorNEE;
                input2.rRec.its = input.rRec.its;

                RadianceQueryRecord &rRec = input2.rRec;
                Intersection &its = rRec.its;
                DirectSamplingRecord dRec(its);

                /* ==================================================================== */
                /*                          Luminaire sampling                          */
                /* ==================================================================== */

                /* Estimate the direct illumination if this is requested */
                if (m_doNee &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bsdfHasSmoothComponent)) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, its, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero())
                    {
                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));

                        Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                        /* Prevent light leaks due to the use of shading normals */
                        if (!m_strictNormals || woDotGeoN * Frame::cosTheta(bRec.wo) > 0) {
                            /* Evaluate BSDF * cos(theta) */
                            const Spectrum bsdfVal = bsdf->eval(bRec);

                            /* Calculate prob. of having generated that direction using BSDF sampling */
                            const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                            Float woPdf = 0, bsdfPdf = 0, dTreePdf = 0;
                            if (emitter->isOnSurface() && dRec.measure == ESolidAngle) {
                                pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, dTree);
                            }

                            /* If we do not use separate splitting factors for guiding/bsdf, misMultGuid == 0 */
                            const Float bsdfMISPDF = splitGuiding ? bsdfPdf : woPdf;

                            /* Weight using the power heuristic */
                            const Float weight = miWeight(dRec.pdf   * misMultNee,
                                                          bsdfMISPDF * misMultBsdf,
                                                          dTreePdf   * misMultGuid);

                            if (!bsdfVal.isZero()) {
                                // we do not account for next event estimation in our irradiance estimate when the bsdf is zero.
                                // otherwise we get massive problems with surfaces that are only visible from one side but are
                                // illuminated strongly on the non-visible side (e.g. the ceiling in the living-room scene).
                                if (rrsMode == EADRRS2) {
                                    irradianceAccumulator += bsdfVal * value * weight / splittingFactorNEE;
                                } else {
                                    irradianceAccumulator += std::abs(Frame::cosTheta(bRec.wo)) * value * weight / splittingFactorNEE;
                                }
                            }

                            value = bsdfVal * value * weight;
                            output.reflected += transmittance * value / splittingFactorNEE;

                            atomicAdd(&m_pathLengthHistogram[rRec.depth], 1);
                            contributionHistogram[rRec.depth] += (value * input2.relativeThroughput).average();

                            if (!m_isFinalIter && m_nee != EAlways) {
                                if (dTree) {
                                    Vertex v = Vertex{
                                        dTree,
                                        dTreeVoxelSize,
                                        Ray(its.p, dRec.d, 0),
                                        bsdfVal / dRec.pdf,
                                        input.relativeThroughput * transmittance / splittingFactorNEE,
                                        bsdfVal,
                                        std::abs(Frame::cosTheta(bRec.wo)),
                                        value,
                                        dRec.pdf,
                                        bsdfPdf,
                                        dTreePdf,
                                        false,
                                    };

                                    v.commit(*m_sdTree, 0.5f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
                                }
                            }
                        } else {
                            value = Spectrum { 0.f };
                        }
                    }

                    output.cost += COST_NEE;
                    if (dTree && !m_isFinalIter) {
                        Float outlierHeuristic = (value * input2.relativeThroughput).average();
                        outlierHeuristic = OUTLIER_MAX == -1 ? 1 :
                                            std::max(Float(1), outlierHeuristic / OUTLIER_MAX);
                        dTree->neeStatistics[binIndex].addSample(
                            input.relativeThroughput.average(),
                            value / outlierHeuristic,
                            COST_NEE
                        );
                    }
                }
            }

            Assert(splitGuiding || numSamplesGuiding == 0);
            for (int sampleIndex = 0; sampleIndex < numSamplesBSDF + numSamplesGuiding; ++sampleIndex) {
                const bool guiding = sampleIndex >= numSamplesBSDF;
                const Float local_splittingFactor = guiding ? splittingFactorGuiding : splittingFactorBSDF;
                const Float local_bsdfSamplingFraction = splitGuiding ? (guiding ? 0.0 : 1.0)
                                                                       : bsdfSamplingFraction;

                LiInput input2 = input;
                input2.breadth *= numSamplesBSDF + numSamplesGuiding;
                input2.throughput *= transmittance / local_splittingFactor;
                input2.relativeThroughput *= transmittance / local_splittingFactor;
                input2.rRec.its = input.rRec.its;

                RadianceQueryRecord &rRec = input2.rRec;
                Intersection &its = rRec.its;
                DirectSamplingRecord dRec(its);

                Spectrum indirectEstimate { 0.f };
                Spectrum directEstimate { 0.f };
                Float nestedCost { 0.f };

                /* ==================================================================== */
                /*                            BSDF sampling                             */
                /* ==================================================================== */

                bool guidedSample = false;
                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                Float woPdf, bsdfPdf, dTreePdf;
                Spectrum bsdfWeight = sampleMat(bsdf, bRec, woPdf, bsdfPdf, dTreePdf, local_bsdfSamplingFraction, rRec, dTree, guidedSample);
                bool isDelta = bRec.sampledType & BSDF::EDelta;
                Float cosTheta = Frame::cosTheta(bRec.wo);
                nestedCost += guidedSample ? COST_GUIDING : COST_BSDF;
                Assert(!isDelta || dTreePdf == 0);

                do {
                    if (bsdfWeight.isZero()) {
                    terminateLocalPath:
                        //uint64_t &l = m_pathLengthHistogram[input.rRec.depth];
                        //reinterpret_cast<std::atomic<uint64_t> &>(l)++;
                        break;
                    }

                    /* Prevent light leaks due to the use of shading normals */
                    const Vector wo = its.toWorld(bRec.wo);
                    Float woDotGeoN = dot(its.geoFrame.n, wo);

                    if (woDotGeoN * cosTheta <= 0 && m_strictNormals)
                        goto terminateLocalPath;

                    /* Trace a ray in this direction */
                    const Float bsdfMISPDF  = splitGuiding ? bsdfPdf : woPdf;
                    input2.ray = Ray(its.p, wo, ray.time);

                    if (its.isMediumTransition())
                        rRec.medium = its.getTargetMedium(input2.ray.d);

                    /* Handle index-matched medium transitions specially */
                    if (bRec.sampledType == BSDF::ENull) {
                        if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                            goto terminateLocalPath;

                        rRec.type = input.scattered ? RadianceQueryRecord::ERadianceNoEmission
                            : RadianceQueryRecord::ERadiance;

                        scene->rayIntersect(input2.ray, its);
                    } else {
                        Spectrum value { 0.0f };
                        rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                            m_maxDepth - rRec.depth - 1, input2.ray, its, dRec, value);

                        /* If a luminaire was hit, estimate the local illumination and
                        weight using the power heuristic */
                        if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) && !value.isZero()) {
                            const Float emitterPdf = (m_doNee && !isDelta) ? scene->pdfEmitterDirect(dRec) : 0;

                            const Float weight = guiding ?
                                miWeight(dTreePdf   * misMultGuid,
                                         bsdfMISPDF * misMultBsdf,
                                         emitterPdf * misMultNee)
                                    :
                                miWeight(bsdfMISPDF * misMultBsdf,
                                         dTreePdf   * misMultGuid,
                                         emitterPdf * misMultNee);

                            directEstimate = bsdfWeight * value * weight;
                            output.reflected += transmittance * directEstimate / local_splittingFactor;

                            atomicAdd(&m_pathLengthHistogram[rRec.depth], 1);
                            contributionHistogram[rRec.depth] += (value * weight * input2.relativeThroughput).average(); /// @todo hack

                            if (rrsMode == EADRRS2) {
                                irradianceAccumulator += bsdfWeight * value * weight / local_splittingFactor;
                            } else {
                                const Float diffuseBsdfWeight = std::abs(Frame::cosTheta(bRec.wo)) / woPdf;
                                irradianceAccumulator += diffuseBsdfWeight * value * weight / local_splittingFactor;
                            }
                        }

                        /// From now on, bsdfWeight is MIS weighted. Can't do earlier because of direct light hits
                        if (splitGuiding) {
                            bsdfWeight *= guiding ?
                                    miWeight(dTreePdf * misMultGuid, bsdfMISPDF * misMultBsdf) :
                                    miWeight(bsdfMISPDF * misMultBsdf, dTreePdf * misMultGuid);
                        } else {
                            // no MIS needed, one-sample MIS is already performed by sampleMat
                        }

                        input2.throughput *= bsdfWeight;
                        input2.relativeThroughput *= bsdfWeight;
                        input2.bsdfPdfCorrection *= (bsdfPdf < 1e-3 ? 1 : woPdf / bsdfPdf);

                        /* Keep track of the throughput, medium, and relative
                        refractive index along the path */
                        input2.eta *= bRec.eta;

                        /* ==================================================================== */
                        /*                         Indirect illumination                        */
                        /* ==================================================================== */

                        /* Stop if indirect illumination was not requested */
                        if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                            goto terminateLocalPath;

                        rRec.type = RadianceQueryRecord::ERadianceNoEmission;
                        input2.scattered = true;
                        input2.liEstimate = dTree ?
                            dTree->getIndirectIrradianceEstimate() * dTreePdf / std::max<Float>(std::abs(cosTheta), 0.1) :
                            Spectrum { 0. }
                        ;
                    }

                    rRec.depth++;

                    LiOutput output2 = Li(input2, stats, contributionHistogram);
                    directEstimate += bsdfWeight * output2.emitted;
                    indirectEstimate = bsdfWeight * output2.reflected;
                    output.depthAcc += output2.depthAcc;
                    output.depthWeight += output2.depthWeight;

                    output.reflected += transmittance * bsdfWeight * output2.totalContribution() / local_splittingFactor;
                    nestedCost += output2.cost;

                    // on glossy surfaces, performing the irradiance estimate can yield extreme amounts of noise.
                    // that's because we want to estimate the diffuse illumination, but we are actually not using
                    // a diffuse distribution for sampling.
                    //if (bsdfWeight.average() > 1e-6 * albedo.average()) {
                        // don't rely on samples that will likely be killed by RR, since those will have high
                        // variance as well

                    if (!isDelta) {
                        /// @todo this is problematic. we are doing fewer estimates than we should!

                        Float diffuseBsdfWeight = std::abs(Frame::cosTheta(bRec.wo)) / woPdf;
                        if (splitGuiding) {
                            diffuseBsdfWeight *= guiding ?
                                miWeight(dTreePdf * misMultGuid, bsdfMISPDF * misMultBsdf) :
                                miWeight(bsdfMISPDF * misMultBsdf, dTreePdf * misMultGuid);
                        }

                        if (rrsMode == EADRRS2) {
                            irradianceAccumulator += bsdfWeight * output2.totalContribution() / local_splittingFactor;
                        } else {
                            irradianceAccumulator += diffuseBsdfWeight * output2.totalContribution() / local_splittingFactor;
                        }
                        indirectIrradianceAccumulator += diffuseBsdfWeight * output2.reflected / local_splittingFactor;
                    }
                    //}
                } while (false);

                output.cost += nestedCost;

                if (dTree && !m_isFinalIter && !isDelta) {
                    Spectrum value = directEstimate + indirectEstimate;
                    Float outlierHeuristic = (value * input2.relativeThroughput).average();
                    outlierHeuristic = OUTLIER_MAX == -1 ? 1 :
                                         std::max(Float(1), outlierHeuristic / OUTLIER_MAX);
                    if (guiding) {
                        dTree->guidingStatistics[binIndex].addSample(
                            input.relativeThroughput.average(),
                            value / outlierHeuristic,
                            nestedCost
                        );
                    } else {
                        if (rrsMode == EClassicRRS && splitGuiding) {
                            /* For training iterations w/o guiding, use BSDF samples to init guiding tree as well. */
                            dTree->guidingStatistics[binIndex].addSample(
                                input.relativeThroughput.average(),
                                value / outlierHeuristic,
                                nestedCost
                            );
                        }
                        dTree->bsdfStatistics[binIndex].addSample(
                            input.relativeThroughput.average(),
                            value / outlierHeuristic,
                            nestedCost
                        );
                    }
                }

                if ((!isDelta || m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) && dTree && !m_isFinalIter) {
                    /* Need to 'recompute' because in the case of `splitGuiding` this has the wrong value */
                    woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * dTreePdf;
                    if (1 / woPdf > 0) {
                        Vertex v {
                            dTree,
                            dTreeVoxelSize,
                            input2.ray,
                            bsdfWeight,
                            input.relativeThroughput * transmittance / local_splittingFactor,
                            bsdfWeight * woPdf,
                            std::abs(cosTheta),
                            indirectEstimate + ((m_nee == EAlways) ? Spectrum{0.0f} : directEstimate),
                            woPdf,
                            bsdfPdf,
                            dTreePdf,
                            isDelta,
                        };

                        v.commit(*m_sdTree, 0.5f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
                    }
                }
            }

            if (dTree && !m_isFinalIter) {
                dTree->irradianceStatistics[binIndex].addSample(irradianceAccumulator, 1);
                dTree->indirectIrradianceStatistics.addSample(indirectIrradianceAccumulator, 1);
            }
        }

        if (output.depthAcc == 0) {
            /// all BSDF samples have failed :-(
            output.markAsLeaf(rRec.depth);
        }

        return output;
    }

    /**
    * This function is called by the recursive ray tracing above after
    * having sampled a direction from a BSDF/phase function. Due to the
    * way in which this integrator deals with index-matched boundaries,
    * it is necessarily a bit complicated (though the improved performance
    * easily pays for the extra effort).
    *
    * This function
    *
    * 1. Intersects 'ray' against the scene geometry and returns the
    *    *first* intersection via the '_its' argument.
    *
    * 2. It checks whether the intersected shape was an emitter, or if
    *    the ray intersects nothing and there is an environment emitter.
    *    In this case, it returns the attenuated emittance, as well as
    *    a DirectSamplingRecord that can be used to query the hypothetical
    *    sampling density at the emitter.
    *
    * 3. If current shape is an index-matched medium transition, the
    *    integrator keeps on looking on whether a light source eventually
    *    follows after a potential chain of index-matched medium transitions,
    *    while respecting the specified 'maxDepth' limits. It then returns
    *    the attenuated emittance of this light source, while accounting for
    *    all attenuation that occurs on the way.
    */
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
        const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
        DirectSamplingRecord &dRec, Spectrum &value) const {
        Intersection its2, *its = &_its;
        Spectrum transmittance(1.0f);
        bool surface = false;
        int interactions = 0;

        while (true) {
            surface = scene->rayIntersect(ray, *its);

            if (medium)
                transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

            if (surface && (interactions == maxInteractions ||
                !(its->getBSDF()->getType() & BSDF::ENull) ||
                its->isEmitter())) {
                /* Encountered an occluder / light source */
                break;
            }

            if (!surface)
                break;

            if (transmittance.isZero())
                return;

            if (its->isMediumTransition())
                medium = its->getTargetMedium(ray.d);

            Vector wo = its->shFrame.toLocal(ray.d);
            BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
            bRec.typeMask = BSDF::ENull;
            transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

            ray.o = ray(its->t);
            ray.mint = Epsilon;
            its = &its2;

            if (++interactions > 100) { /// Just a precaution..
                Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
                return;
            }
        }

        if (surface) {
            /* Intersected something - check if it was a luminaire */
            if (its->isEmitter()) {
                dRec.setQuery(ray, *its);
                value = transmittance * its->Le(-ray.d);
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env && env->fillDirectSamplingRecord(dRec, ray)) {
                value = transmittance * env->evalEnvironment(RayDifferential(ray));
                dRec.dist = std::numeric_limits<Float>::infinity();
                its->t = std::numeric_limits<Float>::infinity();
            }
        }
    }

    Float miWeight(Float pdfA, Float pdfB, Float pdfC=0) const {
        //pdfA *= pdfA; pdfB *= pdfB; pdfC *= pdfC;
        return pdfA / (pdfA + pdfB + pdfC);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "GuidedPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

private:
    mutable EstimatorStatistics<AtomicDouble> m_imageStatistics;
    Spectrum m_imageSqErr { 0.f };
    Float m_imageCost  { 0.f };

    /// Denoising Auxiliary data
    oidn::DeviceRef m_oidnDevice;
    ref<Bitmap> m_denoiseAuxNormals;
    ref<Bitmap> m_denoiseAuxAlbedo;
    ref<Bitmap> m_pixelEstimate;

    /// The datastructure for guiding paths.
    std::unique_ptr<STree> m_sdTree;

    /// The squared values of our currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_squaredImage;
    /// The currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_image;

    std::unique_ptr<StatsImageBlocks> m_statsImages;
    mutable ref<ImageBlock> m_debugImage;
    mutable ref<Film> m_debugFilm;

    std::vector<ref<Bitmap>> m_images;
    std::vector<Float> m_variances;

    /// The modes of NEE which are supported.
    enum ENee {
        ENever,
        EKickstart,
        EAlways,
    };

    /**
        How to perform next event estimation (NEE). The following values are valid:
        - "never":     Never performs NEE.
        - "kickstart": Performs NEE for the first few iterations to initialize
                       the SDTree with good direct illumination estimates.
        - "always":    Always performs NEE.
        Default = "never"
    */
    std::string m_neeStr;
    ENee m_nee;

    std::string m_rrsStr;
    enum ERRS {
        ENoRRS,     // no RR or splitting
        EAlbedoRR,  // RR based on throughput, but divided by BSDF PDF instead of MIS PDF
        EAlbedo2RRS, // RR based on BSDF albedo
        EClassicRRS, // classical RR based on throughput, no splitting
        EADRRS,     // adjoint-driven russian roulette and splitting
        EADRRS2,    // adjoint-driven russian roulette and splitting, without diffuse irradiance assumption
        EMARS       // efficiency-aware russian roulette and splitting
    };
    ERRS m_rrsMode;
    bool m_useAbsoluteMetric;
    bool m_addAdjointEstimate;

    Float m_splittingMin;
    Float m_splittingMax;
    Float m_branchingMax;

    MARSSplitConfig m_splitConfig;

    /// Whether Li should currently perform NEE (automatically set during rendering based on m_nee).
    bool m_doNee;

    enum EBudget {
        ESpp,
        ESeconds,
    };

    /**
        What type of budget to use. The following values are valid:
        - "spp":     Budget is the number of samples per pixel.
        - "seconds": Budget is a time in seconds.
        Default = "seconds"
    */
    std::string m_budgetStr;
    EBudget m_budgetType;
    Float m_budget;

    Float m_trainingBudget;
    Float m_renderingBudget;

    bool m_isBuilt = false;
    int m_iter;
    bool m_isFinalIter = false;

    int m_sppPerPass;

    int m_passesRendered;
    int m_passesRenderedThisIter;
    mutable std::unique_ptr<ProgressReporter> m_progress;

    std::vector<ref<BlockedRenderProcess>> m_renderProcesses;

    /**
        How to combine the samples from all path-guiding iterations:
        - "discard":    Discard all but the last iteration.
        - "automatic":  Discard all but the last iteration, but automatically assign an appropriately
                        larger budget to the last [Mueller et al. 2018].
        - "inversevar": Combine samples of the last 4 iterations based on their
                        mean pixel variance [Mueller et al. 2018].
        Default     = "automatic" (for reproducibility)
        Recommended = "inversevar"
    */
    std::string m_sampleCombinationStr;
    ESampleCombination m_sampleCombination;


    /// Maximum memory footprint of the SDTree in MB. Stops subdividing once reached. -1 to disable.
    int m_sdTreeMaxMemory;

    /**
        The spatial filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "stochastic": Stochastic box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        - "box":        Box filter; improves the quality further at significant
                        additional computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "stochastic"
    */
    std::string m_spatialFilterStr;
    ESpatialFilter m_spatialFilter;

    /**
        The directional filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "box":        Box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "box"
    */
    std::string m_directionalFilterStr;
    EDirectionalFilter m_directionalFilter;

    /**
        Leaf nodes of the spatial binary tree are subdivided if the number of samples
        they received in the last iteration exceeds c * sqrt(2^k) where c is this value
        and k is the iteration index. The first iteration has k==0.
        Default     = 12000 (for reproducibility)
        Recommended = 4000
    */
    int m_sTreeThreshold;

    /**
        Leaf nodes of the directional quadtree are subdivided if the fraction
        of energy they carry exceeds this value.
        Default = 0.01 (1%)
    */
    Float m_dTreeThreshold;

    /**
        When guiding, we perform MIS with the balance heuristic between the guiding
        distribution and the BSDF, combined with probabilistically choosing one of the
        two sampling methods. This factor controls how often the BSDF is sampled
        vs. how often the guiding distribution is sampled.
        Default = 0.5 (50%)
    */
    Float m_bsdfSamplingFraction;

    /**
        The loss function to use when learning the bsdfSamplingFraction using gradient
        descent, following the theory of Neural Importance Sampling [Mueller et al. 2018].
        The following values are valid:
        - "none":  No learning (uses the fixed `m_bsdfSamplingFraction`).
        - "kl":    Optimizes bsdfSamplingFraction w.r.t. the KL divergence.
        - "var":   Optimizes bsdfSamplingFraction w.r.t. variance.
        Default     = "none" (for reproducibility)
        Recommended = "kl"
    */
    std::string m_bsdfSamplingFractionLossStr;
    EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss;

    /**
        Whether to dump a binary representation of the SD-Tree to disk after every
        iteration. The dumped SD-Tree can be visualized with the accompanying
        visualizer tool.
        Default = false
    */
    bool m_dumpSDTree;
    std::string m_sdTreePath;

    int m_trainingIterations;
    bool m_disableGuiding;

    mutable std::vector<int64_t> m_pathLengthHistogram;
    mutable std::vector<double> m_contributionHistogram;

    /// The time at which rendering started.
    std::chrono::steady_clock::time_point m_startTime;

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(GuidedPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathTracer, "Guided path tracer");
MTS_NAMESPACE_END
