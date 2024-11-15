#pragma once

#include <mitsuba/mitsuba.h>

#include <mutex>

#include "FLAGS.h"
#include "octtree.h"

MTS_NAMESPACE_BEGIN

/// Needed to do globally, otherwise p_thread complains
extern std::mutex m_averageMutex;
struct BMARSConfiguration;

struct RRSMethod {
public:
    enum {
        EClassic,
        EBDPT,
        EEARS,
        EBMARS,
    } technique;

    Float splittingMin;
    Float splittingMax;

    int rrDepth;
    bool useAbsoluteThroughput;
    /* Allows reusing the same light path for all camera path intersections.
       Does only work when NOT splitting light paths. */
    bool bmarsReuseLP;

    RRSMethod();
    RRSMethod(
        const Float splittingMin,
        const Float splittingMax,
        const int rrDepth,
        std::string &rrsStr,
        const bool bmarsReuseLP);

    static RRSMethod Classic();
    static RRSMethod BDPT();

    std::string getName() const;

    int maxEmitterDepth() const;

    Float evaluate(
        const Octtree::SamplingNode *samplingNode,
        Float imageEarsFactor,
        const Spectrum &albedo,
        const Spectrum &throughput,
        bool bsdfHasSmoothComponent,
        int depth,
        Octtree::SplitType st
    ) const;

    bool needsTrainingPhase() const;

    bool performsInvVarWeighting() const;

    bool needsPixelEstimate() const;

private:
    Float clamp(Float splittingFactor) const;

    Float weightWindow(Float splittingFactor, Float weightWindowSize = 5) const;
};

/**
 * Helper class to build averages that discared a given amount of outliers.
 * Used for our image variance estimate.
 */
class OutlierRejectedAverage {
public:
    struct Sample {
        Spectrum secondMoment;
        Float cost;

        Sample();

        Sample(const Spectrum &sm, Float cost);

        void operator+=(const Sample &other);
        void operator-=(const Sample &other);
        Sample operator-(const Sample &other) const;
        Sample operator/(Float weight) const;

        bool operator>=(const Sample &other) const;
    };

    /**
     * Resizes the history buffer to account for up to \c length outliers.
     */
    void resize(int length);

    /**
     * Resets all statistics, including outlier history and current average.
     */
    void reset();

    /**
     * Returns whether a lower bound can be given on what will definitely not count as outlier.
     */
    bool hasOutlierLowerBound() const;

    /**
     * Returns the lower bound of what will definitely count as outlier.
     * Useful if multiple \c OutlierRejectedAverage from different threads will be combined.
     */
    Sample outlierLowerBound() const;

    /**
     * Sets a manual lower bound of what will count as outlier.
     * This avoids wasting time on adding samples to the outlier history that are known to be less significant
     * than outliers that have already been collected by other instances of \c OutlierRejectedAverage that
     * will eventually be merged.
     */
    void setRemoteOutlierLowerBound(const Sample &minimum);

    /**
     * Records one sample.
     */
    void operator+=(Sample sample);

    /**
     * Merges the statistics of another \c OutlierRejectedAverage into this instance.
     */
    void operator+=(const OutlierRejectedAverage &other);

    void dump() const;

    void computeOutlierContribution();

    Sample average() const;
    Sample averageWithoutRejection() const;

    long weight() const;

private:
    long m_weight;
    int m_index;
    int m_length;
    Sample m_accumulation;
    Sample m_knownMinimum;
    std::vector<Sample> m_history;

    Sample m_outlierAccumulation;
    long m_outlierWeight;
};

struct ImageStatistics {
    void setOutlierRejectionCount(int count);
    void applyOutlierRejection();

    Spectrum squareError() const;

    Float earsFactor() const;
    Float cost() const;
    Float efficiency() const;

    void reset(Float actualTotalCost);


    void
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
    __attribute__((no_sanitize("thread")))
#  endif
#endif
    operator+=(const OutlierRejectedAverage &blockStatistics);

    void splatDepthAcc(Float depthAcc, Float depthWeight, Float primarySplit, Float primarySamples, Float neeSplit, Float lpSplit);

    bool hasOutlierLowerBound() const;
    OutlierRejectedAverage::Sample outlierLowerBound() const;

private:
    OutlierRejectedAverage m_average;
    Float m_depthAcc { 0.f };
    Float m_depthWeight { 0.f };
    Float m_primarySplit { 0.f };
    Float m_primarySamples { 0.f };
    Float m_neeSplit { 0.f };
    Float m_lpSplit { 0.f };

    struct {
        Spectrum squareError;
        Float cost;
    } m_lastStats;
};

MTS_NAMESPACE_END
