#pragma once

#include <mitsuba/mitsuba.h>

#include <mutex>

#include "FLAGS.h"

MTS_NAMESPACE_BEGIN

/// Needed to do globally, otherwise p_thread complains
extern std::mutex m_averageMutex;
struct EffMISContext;

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

MTS_NAMESPACE_END
