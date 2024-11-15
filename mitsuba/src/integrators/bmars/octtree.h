#pragma once

#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <vector>

#include "FLAGS.h"

MTS_NAMESPACE_BEGIN

void atomicAdd(Spectrum *dest, const Spectrum &src);

class Octtree {
public:
    enum SplitType {
        LR,
        NEE,
        LP,
        STCount   /// End dummy to iterate and get count. Must be the last entry
    };

    static constexpr int HISTOGRAM_RESOLUTION = 4;
    static constexpr int BIN_COUNT = HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION;

    struct Configuration {
        Float minimumLeafWeightForSampling = 40000;
        Float minimumLeafWeightForTraining = 20000;
        Float leafDecay = 0; /// set to 0 for hard reset after an iteration, 1 for no reset at all
        long maxNodeCount = 0;
        bool shareSF = false;
    };

    struct TrainingNode {
        TrainingNode() {
            m_FirstMoment.fill(Spectrum { 0. });
            m_SecondMoment.fill(Spectrum { 0. });
        }

        void decay(Float decayFactor) {
            for (int i = 0; i < STCount; ++i) {
                m_Weight[i] *= decayFactor;
                m_FirstMoment[i] *= decayFactor;
                m_SecondMoment[i] *= decayFactor;
                m_Cost[i] *= decayFactor;
            }
        }

        TrainingNode &operator+=(const TrainingNode &other) {
            for (int i = 0; i < STCount; ++i) {
                m_Weight[i] += other.m_Weight[i];
                m_FirstMoment[i] += other.m_FirstMoment[i];
                m_SecondMoment[i] += other.m_SecondMoment[i];
                m_Cost[i] += other.m_Cost[i];
            }
            return *this;
        }

        Float getWeight(SplitType st) const {
            return m_Weight[st];
        }

        Float getCombinedWeight() const {
            Float result = 0;
            for (int i = 0; i < STCount; ++i) {
                const SplitType st = static_cast<SplitType>(i);
                result = std::max(result, m_Weight[st]);
            }
            return result;
        }

        Spectrum getEstimate(SplitType st) const {
            return m_Weight[st] > 0 ? (m_FirstMoment[st] / m_Weight[st]) : Spectrum { 0. };
        }

        Spectrum getSecondMoment(SplitType st) const {
            return m_Weight[st] > 0 ? m_SecondMoment[st] / m_Weight[st] : Spectrum { 0. };
        }

        Spectrum getVariance(SplitType st) const {
            if (m_Weight[st] == 0)
                return Spectrum(0.f);

            Spectrum result;
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
                result[i] = std::max(m_SecondMoment[st][i] / m_Weight[st] - Float(std::pow(m_FirstMoment[st][i] / m_Weight[st], 2)), Float(0));
            }
            return result;
        }

        Float getCost(SplitType st) const {
            return m_Weight[st] > 0 ? m_Cost[st] / m_Weight[st] : 0;
        }

        void splatEstimate(const Spectrum &sum, const Spectrum &sumSquares, Float cost, Float weight, SplitType st) {
            atomicAdd(&m_FirstMoment[st], sum);
            atomicAdd(&m_SecondMoment[st], sumSquares);
            atomicAdd(&m_Cost[st], cost);
            atomicAdd(&m_Weight[st], weight);
        }

    private:
        std::array<Float, STCount> m_Weight {};
        std::array<Spectrum, STCount> m_FirstMoment;
        std::array<Spectrum, STCount> m_SecondMoment;
        std::array<Float, STCount> m_Cost {};
    };

    struct SamplingNode {
        SamplingNode() {
            estimate.fill(Spectrum { 0. });
            earsFactorR.fill(Spectrum { 0. });
            earsFactorS.fill(Spectrum { 0. });
        }

        bool isValid(SplitType st) const { return m_isValid[st]; }
        void learnFrom(const TrainingNode &trainingNode, const Configuration &config) {
            // For sharing ONE splitting factor, just put the whole estimate into LR cache.
            if (config.shareSF) {
                m_isValid[SplitType::LR] = std::max<Float>(std::max<Float>(
                                                trainingNode.getWeight(SplitType::LR),
                                                trainingNode.getWeight(SplitType::NEE)),
                                                trainingNode.getWeight(SplitType::LP))
                                            >= config.minimumLeafWeightForSampling;

                if (trainingNode.getWeight(SplitType::LR) > 0) {
                    auto scndMom = trainingNode.getSecondMoment(SplitType::LR)
                                    + trainingNode.getSecondMoment(SplitType::NEE)
                                    + 2 * trainingNode.getEstimate(SplitType::LR) * trainingNode.getEstimate(SplitType::NEE)
                                    + trainingNode.getSecondMoment(SplitType::LP)
                                    + 2 * trainingNode.getEstimate(SplitType::LP) * trainingNode.getEstimate(SplitType::LR)
                                    + 2 * trainingNode.getEstimate(SplitType::LP) * trainingNode.getEstimate(SplitType::NEE);
                    auto variance = trainingNode.getVariance(SplitType::LR)
                                    + trainingNode.getVariance(SplitType::NEE)
                                    + trainingNode.getVariance(SplitType::LP);
                    const auto cost = trainingNode.getCost(SplitType::LR)
                                      + trainingNode.getCost(SplitType::NEE)
                                      + trainingNode.getCost(SplitType::LP);
                    earsFactorR[SplitType::LR] = scndMom / cost;
                    earsFactorS[SplitType::LR] = variance / cost;
                }
            } else {
                for (int i = 0; i < STCount; ++i) {
                    const SplitType st = static_cast<SplitType>(i);
                    m_isValid[st] = trainingNode.getWeight(st) >= config.minimumLeafWeightForSampling;

                    if (trainingNode.getWeight(st) > 0) {
                        estimate[st] = trainingNode.getEstimate(st);

                        if (trainingNode.getCost(st) > 0) {
                            earsFactorR[st] = trainingNode.getSecondMoment(st) / trainingNode.getCost(st);
                            earsFactorS[st] = trainingNode.getVariance(st) / trainingNode.getCost(st);
                        } else {
                            /// there can be caches where no work is done
                            /// (e.g., failed strict normals checks meaning no NEE samples or BSDF samples are ever taken)
                            earsFactorR[st] = Spectrum { 0.f };
                            earsFactorS[st] = Spectrum { 0.f };
                        }
                    }
                }
            }
        }

        std::array<Spectrum, STCount> estimate;
        std::array<Spectrum, STCount> earsFactorR; // sqrt(2nd-moment / cost)
        std::array<Spectrum, STCount> earsFactorS; // sqrt(variance / cost)

    private:
        std::array<bool, STCount> m_isValid;
    };

    Configuration configuration;

    void setMaximumMemory(long bytes) {
        configuration.maxNodeCount = (bytes * STCount) / sizeof(Node);
    }
    void setShareSF(bool s) {
        configuration.shareSF = s;
    }
    void setLeafDecay(Float ld) {
        configuration.leafDecay = ld;
    }

private:
    typedef uint32_t NodeIndex;

    struct Node {
        struct Child {
            NodeIndex index { 0 };
            std::array<TrainingNode, BIN_COUNT> training;
            std::array<SamplingNode, BIN_COUNT> sampling;

            bool isLeaf() const { return index == 0; }
            Float maxTrainingWeight() const {
                Float weight = 0;
                for (const auto &t : training)
                    weight = std::max(weight, t.getCombinedWeight());
                return weight;
            }
        };

        std::array<Child, 8> children;
    };

    std::vector<Node> m_nodes;

    int stratumIndex(Vector3 &pos) {
        int index = 0;
        for (int dim = 0; dim < 3; ++dim) {
            int bit = pos[dim] >= 0.5f;
            index |= bit << dim;
            pos[dim] = pos[dim] * 2 - bit;
        }
        return index;
    }

    NodeIndex splitNodeIfNecessary(Float weight) {
        if (weight < configuration.minimumLeafWeightForTraining)
            /// splitting not necessary
            return 0;

        if (configuration.maxNodeCount && long(m_nodes.size()) > configuration.maxNodeCount)
            /// we have already reached the maximum node number
            return 0;

        NodeIndex newNodeIndex = NodeIndex(m_nodes.size());
        m_nodes.emplace_back();

        for (int stratum = 0; stratum < 8; ++stratum) {
            /// split recursively if needed
            NodeIndex newChildIndex = splitNodeIfNecessary(weight / 8);
            m_nodes[newNodeIndex].children[stratum].index = newChildIndex;
        }

        return newNodeIndex;
    }

    std::array<TrainingNode, BIN_COUNT> build(NodeIndex index, bool needsSplitting) {
        std::array<TrainingNode, BIN_COUNT> sum;

        for (int stratum = 0; stratum < 8; ++stratum) {
            if (m_nodes[index].children[stratum].isLeaf()) {
                if (needsSplitting) {
                    NodeIndex newChildIndex = splitNodeIfNecessary(
                        m_nodes[index].children[stratum].maxTrainingWeight()
                    );
                    m_nodes[index].children[stratum].index = newChildIndex;
                }
            } else {
                /// build recursively
                auto buildResult = build(
                    m_nodes[index].children[stratum].index,
                    needsSplitting
                );
                m_nodes[index].children[stratum].training = buildResult;
            }

            auto &child = m_nodes[index].children[stratum];
            for (int bin = 0; bin < BIN_COUNT; ++bin) {
                sum[bin] += child.training[bin];
                child.sampling[bin].learnFrom(child.training[bin], configuration);
                child.training[bin].decay(configuration.leafDecay);
            }
        }

        return sum;
    }

public:
    Octtree() {
        m_nodes.emplace_back();

        /// initialize tree to some depth
        for (int stratum = 0; stratum < 8; ++stratum) {
            NodeIndex newChildIndex = splitNodeIfNecessary(
                8 * configuration.minimumLeafWeightForSampling
            );
            m_nodes[0].children[stratum].index = newChildIndex;
        }
    }

    /**
     * Accumulates all the data from training into the sampling nodes, refines the tree and resets the training nodes.
     */
    void build(bool needsSplitting) {
        auto sum = build(0, needsSplitting);
        m_nodes.shrink_to_fit();

        Float weightSum = 0;
        for (int bin = 0; bin < BIN_COUNT; ++bin)
            weightSum += sum[bin].getCombinedWeight();

        SLog(EInfo, "Octtree built [%ld samples, %ld nodes, %.1f MiB]",
            long(weightSum),
            m_nodes.size(),
            m_nodes.capacity() * sizeof(Node) / (1024.f * 1024.f)
        );
    }

    void lookup(Vector3 pos, int bin, const SamplingNode* &sampling, TrainingNode* &training, SplitType st) {
        NodeIndex currentNodeIndex = 0;
        while (true) {
            int stratum = stratumIndex(pos);
            auto &child = m_nodes[currentNodeIndex].children[stratum];
            if (currentNodeIndex == 0 || child.sampling[bin].isValid(st))
                /// a valid node for sampling
                sampling = &child.sampling[bin];

            if (child.isLeaf()) {
                /// reached a leaf node
                training = &child.training[bin];
                break;
            }

            currentNodeIndex = child.index;
        }
    }
};

MTS_NAMESPACE_END
