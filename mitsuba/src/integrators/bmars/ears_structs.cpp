#include "ears_structs.h"

#include "bmars.h"

MTS_NAMESPACE_BEGIN

std::mutex m_averageMutex;

/* ==================================================================== */
/*                               RRSMethod                              */
/* ==================================================================== */


RRSMethod::RRSMethod() {
    technique = EClassic;
    splittingMin = 1;
    splittingMax = 1;
    rrDepth = 1;
    useAbsoluteThroughput = true;
}

RRSMethod::RRSMethod(
    const Float splittingMin,
    const Float splittingMax,
    const int rrDepth,
    std::string &rrsStr,
    const bool bmarsReuseLP
        ) :
    splittingMin(splittingMin),
    splittingMax(splittingMax),
    rrDepth(rrDepth),
    bmarsReuseLP(bmarsReuseLP)
{
    /// parse desired modifiers
    if ((useAbsoluteThroughput = rrsStr.back() == 'A')) {
        rrsStr.pop_back();
    }

    if (rrsStr.back() == 'S') {
        rrsStr.pop_back();
    } else if (this->splittingMax > 1) {
        SLog(EWarn, "Changing maximum splitting factor to 1 since splitting was not explicitly allowed");
        this->splittingMax = 1;
    }

    /// parse desired technique
    if (rrsStr == "BMAR")      technique = EBMARS; else
    if (rrsStr == "EAR")       technique = EEARS; else
    if (rrsStr == "bdptRR")    technique = EBDPT; else
    if (rrsStr == "classicRR") technique = EClassic; else {
        SLog(EError, "Invalid RRS technique specified: %s", rrsStr.c_str());
    }

    if ((technique == EBMARS || technique == EEARS) && rrDepth != 1)
        SLog(EWarn, "MARS (or EARS) should ideally be used with rrDepth 1");

    SAssert((technique != EBMARS && technique != EEARS) || splittingMin != 0);
}

RRSMethod RRSMethod::Classic() {
    RRSMethod rrs;
    rrs.technique    = EClassic;
    rrs.splittingMin = 0;
    rrs.splittingMax = 0.95;
    rrs.rrDepth      = 5;
    rrs.useAbsoluteThroughput = true;
    rrs.bmarsReuseLP = false;
    return rrs;
}

RRSMethod RRSMethod::BDPT() {
    RRSMethod rrs;
    rrs.technique    = EBDPT;
    rrs.splittingMin = 0.05;
    rrs.splittingMax = 0.95;
    rrs.rrDepth      = 5;
    rrs.useAbsoluteThroughput = true;
    rrs.bmarsReuseLP = false;
    return rrs;
}

int RRSMethod::maxEmitterDepth() const {
    switch (technique) {
        case EClassic: return 1;
        case EBDPT:    return std::numeric_limits<int>::max();
        case EEARS:    return 1;
        case EBMARS:   return std::numeric_limits<int>::max();
    }
    /// make gcc happy
    return 0;
}

std::string RRSMethod::getName() const {
    std::string suffix = "";
    if (splittingMax > 1) suffix += "S";
    if (useAbsoluteThroughput) suffix += "A";

    switch (technique) {
        case EClassic: return "classicRR" + suffix;
        case EBDPT:    return "bdptRR" + suffix;
        case EEARS:    return "EAR" + suffix;
        case EBMARS:   return "BMAR" + suffix;
        default:       return "ERROR";
    }
}

Float RRSMethod::evaluate(
    const Octtree::SamplingNode *samplingNode,
    Float imageEarsFactor,
    const Spectrum &albedo,
    const Spectrum &throughput,
    bool bsdfHasSmoothComponent,
    int depth,
    Octtree::SplitType st
) const {
    depth = depth - 1;  // We have an additional edge (=depth) that is not present in EARS
    if (depth < rrDepth) {
        /// do not perform RR or splitting at this depth.
        return 1;
    }

    switch (technique) {
    case EBDPT: {
        /// In BDPT, we only apply RR to LR. NEE and LP splitting factor evaluation should always return 1
        if (st != Octtree::SplitType::LR)
            return clamp(1);

        /// To perfectly imitate mitsuba's BDPT implementation, we need to use `clamp(throughput.max());`
        if (!useAbsoluteThroughput)
            return clamp(throughput.max());

        /// Classic RR(S) based on throughput weight
        if (albedo.isZero())
            /// avoid bias for materials that might report their reflectance incorrectly
            return clamp(0.1f);
        return clamp((throughput * albedo).average());
    }
    case EClassic: {
        /// Classic RR(S) based on throughput weight
        if (albedo.isZero())
            /// avoid bias for materials that might report their reflectance incorrectly
            return clamp(0.1f);
        /// To perfectly imitate mitsuba's BDPT implementation, use `clamp(throughput.max());` instead
        return clamp((throughput * albedo).average());
    }

    case EEARS:
    case EBMARS: {
        /// "Efficiency-Aware Russian Roulette and Splitting"
        if (bsdfHasSmoothComponent) {
            const Float splittingFactorS = std::sqrt( (throughput * throughput * samplingNode->earsFactorS[st]).average() ) * imageEarsFactor;
            const Float splittingFactorR = std::sqrt( (throughput * throughput * samplingNode->earsFactorR[st]).average() ) * imageEarsFactor;

            if (splittingFactorR > 1) {
                if (splittingFactorS < 1) {
                    /// second moment and variance disagree on whether to split or RR, resort to doing nothing.
                    return clamp(1);
                } else {
                    /// use variance only if both modes recommend splitting.
                    return clamp(splittingFactorS);
                }
            } else {
                /// use second moment only if it recommends RR.
                return clamp(splittingFactorR);
            }
        } else if (st == Octtree::SplitType::LR) {
            return clamp(1);
        } else {
            return clamp(0);
        }
    }
    }
    std::cout << technique << std::endl;
    SAssert(false);

    /// make gcc happy
    return 0;
}

bool RRSMethod::needsTrainingPhase() const {
    switch (technique) {
        case EClassic: return false;
        case EBDPT:    return false;
        case EEARS:    return true;
        case EBMARS:   return true;
    }

    // make gcc happy
    return 0;
}

bool RRSMethod::performsInvVarWeighting() const {
    return needsTrainingPhase();
}

bool RRSMethod::needsPixelEstimate() const {
    return useAbsoluteThroughput == false;
}

Float RRSMethod::clamp(Float splittingFactor) const {
    /// not using std::clamp here since that's C++17
    splittingFactor = std::min(splittingFactor, splittingMax);
    splittingFactor = std::max(splittingFactor, splittingMin);
    return splittingFactor;
}

Float RRSMethod::weightWindow(Float splittingFactor, Float weightWindowSize) const {
    const float dminus = 2 / (1 + weightWindowSize);
    const float dplus = dminus * weightWindowSize;

    if (splittingFactor < dminus) {
        /// russian roulette
        return splittingFactor / dminus;
    } else if (splittingFactor > dplus) {
        /// splitting
        return splittingFactor / dplus;
    } else {
        /// within weight window
        return 1;
    }
}




/* ==================================================================== */
/*                         OutlierRejectedAverage                       */
/* ==================================================================== */

OutlierRejectedAverage::Sample::Sample() : secondMoment(Spectrum(0.f)), cost(0) {}
OutlierRejectedAverage::Sample::Sample(const Spectrum &sm, Float cost)
        : secondMoment(sm), cost(cost) {}

void OutlierRejectedAverage::Sample::operator+=(const Sample &other) {
    secondMoment += other.secondMoment;
    cost += other.cost;
}

void OutlierRejectedAverage::Sample::operator-=(const Sample &other) {
    secondMoment -= other.secondMoment;
    cost -= other.cost;
}

OutlierRejectedAverage::Sample OutlierRejectedAverage::Sample::operator-(const Sample &other) const {
    Sample s = *this;
    s -= other;
    return s;
}

OutlierRejectedAverage::Sample OutlierRejectedAverage::Sample::operator/(Float weight) const {
    return Sample {
        secondMoment / weight,
        cost / weight
    };
}

bool OutlierRejectedAverage::Sample::operator>=(const Sample &other) const {
    return secondMoment.average() >= other.secondMoment.average();
}


void OutlierRejectedAverage::resize(int length) {
    m_length = length;
    m_history.resize(length);
    reset();
}

void OutlierRejectedAverage::reset() {
    m_index = 0;
    m_knownMinimum = Sample();
    m_accumulation = Sample();
    m_weight = 0;
    m_outlierAccumulation = Sample();
    m_outlierWeight = 0;
}

bool OutlierRejectedAverage::hasOutlierLowerBound() const {
    return m_length > 0 && m_index >= m_length;
}

OutlierRejectedAverage::Sample OutlierRejectedAverage::outlierLowerBound() const {
    return m_history[m_index - 1];
}

void OutlierRejectedAverage::setRemoteOutlierLowerBound(const Sample &minimum) {
    m_knownMinimum = minimum;
}

void OutlierRejectedAverage::operator+=(Sample sample) {
    m_weight += 1;
    m_accumulation += sample;

    if (m_knownMinimum >= sample) {
        return;
    }

    int insertionPoint = m_index;

    while (insertionPoint > 0 && sample >= m_history[insertionPoint - 1]) {
        if (insertionPoint < m_length) {
            m_history[insertionPoint] = m_history[insertionPoint - 1];
        }
        insertionPoint--;
    }

    if (insertionPoint < m_length) {
        m_history[insertionPoint] = sample;
        if (m_index < m_length) {
            ++m_index;
        }
    }
}

void OutlierRejectedAverage::operator+=(const OutlierRejectedAverage &other) {
    int m_writeIndex = m_index + other.m_index;
    int m_readIndexLocal = m_index - 1;
    int m_readIndexOther = other.m_index - 1;

    while (m_writeIndex > 0) {
        Sample sample;
        if (m_readIndexOther < 0 || (m_readIndexLocal >= 0 && other.m_history[m_readIndexOther] >= m_history[m_readIndexLocal])) {
            /// we take the local sample next
            sample = m_history[m_readIndexLocal--];
        } else {
            /// we take the other sample next
            sample = other.m_history[m_readIndexOther--];
        }

        if (--m_writeIndex < m_length) {
            m_history[m_writeIndex] = sample;
        }
    }

    m_index = std::min(m_index + other.m_index, m_length);
    m_weight += other.m_weight;
    m_accumulation += other.m_accumulation;
}

void OutlierRejectedAverage::dump() const {
    std::cout << m_index << " vs " << m_length << std::endl;
    for (int i = 0; i < m_index; ++i)
        std::cout << m_history[i].secondMoment.average() << std::endl;
}

void OutlierRejectedAverage::computeOutlierContribution() {
    for (int i = 0; i < m_index; ++i) {
        m_outlierAccumulation += m_history[i];
    }
    m_outlierWeight += m_index;

    /// reset ourselves
    m_index = 0;
}

OutlierRejectedAverage::Sample OutlierRejectedAverage::average() const {
    if (m_index > 0) {
        SLog(EWarn, "There are some outliers that have not yet been removed. Did you forget to call computeOutlierContribution()?");
    }

    return (m_accumulation - m_outlierAccumulation) / (m_weight - m_outlierWeight);
}

OutlierRejectedAverage::Sample OutlierRejectedAverage::averageWithoutRejection() const {
    return m_accumulation / m_weight;
}

long OutlierRejectedAverage::weight() const {
    return m_weight;
}


/* ==================================================================== */
/*                             ImageStatistics                          */
/* ==================================================================== */

void ImageStatistics::setOutlierRejectionCount(int count) {
    m_average.resize(count);
}

void ImageStatistics::applyOutlierRejection() {
    m_average.computeOutlierContribution();
}

Float ImageStatistics::cost() const {
    return m_lastStats.cost;
}

Spectrum ImageStatistics::squareError() const {
    return m_lastStats.squareError;
}

Float ImageStatistics::earsFactor() const {
    // SLog(EInfo, "cost: %g\t err: %g", cost(), squareError().average());
    return std::sqrt( cost() / squareError().average() );
}

Float ImageStatistics::efficiency() const {
    return 1 / (cost() * squareError().average());
}

void ImageStatistics::reset(Float actualTotalCost) {
    auto weight = m_average.weight();
    auto avgNoReject = m_average.averageWithoutRejection();
    auto avg = m_average.average();
    m_lastStats.squareError = avg.secondMoment;
    m_lastStats.cost = avg.cost;

    //m_average.dump();
    m_average.reset();

    const Float eF = earsFactor();
    const Float earsFactorNoReject = std::sqrt( avgNoReject.cost / avgNoReject.secondMoment.average() );

    SLog(EInfo, "Averages:\n"
        "  Average path count:  %.3f\n"
        "  Average path length: %.3f\n"
        "  Average LR split:  %.3f\n"
        "  Average NEE split: %.3f\n"
        "  Average LP split:  %.3f\n",
        m_primarySamples > 0 ? m_depthWeight / m_primarySamples : 0,
        m_depthWeight > 0 ? m_depthAcc / m_depthWeight : 0,
        m_primarySamples > 0 ? m_primarySplit / m_primarySamples : 0,
        m_neeSplit > 0 ? m_neeSplit / m_primarySamples : 0,
        m_lpSplit > 0 ? m_lpSplit / m_primarySamples : 0);

    SLog(EInfo, "Statistics:\n"
        "  (values in brackets are without outlier rejection)\n"
        "  Estimated Cost    = %.3e (%.3e)\n"
        "  Actual Cost       = %.3e (  n. a.  )\n"
        "  Variance per SPP  = %.3e (%.3e)\n"
        "  Est. Cost per SPP = %.3e (%.3e)\n"
        "  Est. Efficiency   = %.3e (%.3e)\n"
        "  Act. Efficiency   = %.3e (%.3e)\n"
        "  EARS multiplier   = %.3e (%.3e)\n",
        avg.cost * weight, avgNoReject.cost * weight,
        actualTotalCost,
        squareError().average(), avgNoReject.secondMoment.average(),
        cost(), avgNoReject.cost,
        efficiency(), 1 / (avgNoReject.cost * avgNoReject.secondMoment.average()),
        1 / (actualTotalCost / weight * squareError().average()), 1 / (actualTotalCost / weight * avgNoReject.secondMoment.average()),
        eF, earsFactorNoReject
    );

    m_depthAcc = 0;
    m_depthWeight = 0;
    m_primarySplit = 0;
    m_primarySamples = 0;
    m_neeSplit = 0;
    m_lpSplit = 0;

}

void ImageStatistics::operator+=(const OutlierRejectedAverage &blockStatistics) {
    std::lock_guard<std::mutex> lock(m_averageMutex);
    m_average += blockStatistics;
}

void ImageStatistics::splatDepthAcc(Float depthAcc, Float depthWeight, Float primarySplit, Float primarySamples, Float neeSplit, Float lpSplit) {
    atomicAdd(&m_depthAcc, depthAcc);
    atomicAdd(&m_depthWeight, depthWeight);
    atomicAdd(&m_primarySplit, primarySplit);
    atomicAdd(&m_primarySamples, primarySamples);
    atomicAdd(&m_neeSplit, neeSplit);
    atomicAdd(&m_lpSplit, lpSplit);
}

bool ImageStatistics::hasOutlierLowerBound() const {
    return m_average.hasOutlierLowerBound();
}

OutlierRejectedAverage::Sample ImageStatistics::outlierLowerBound() const {
    return m_average.outlierLowerBound();
}

MTS_NAMESPACE_END
