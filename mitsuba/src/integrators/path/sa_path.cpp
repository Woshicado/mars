#include "GitSHA1.h"

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
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>

/// we support outputting several AOVs that can be helpful for research and debugging.
/// since they are computationally expensive, we disable them by default.
/// uncomment the following line to enable outputting AOVs:
#define EARS_INCLUDE_AOVS

#include "sa_path_aovs.h"

MTS_NAMESPACE_BEGIN

thread_local StatsRecursiveImageBlockCache *StatsRecursiveImageBlockCache::instance = nullptr;
thread_local StatsRecursiveDescriptorCache *StatsRecursiveDescriptorCache::instance = nullptr;
thread_local StatsRecursiveValuesCache *StatsRecursiveValuesCache::instance = nullptr;

// Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer {
public:
    AdamOptimizer(Float learningRate, Float epsilon = 1e-08f, Float beta1 = 0.9f, Float beta2 = 0.999f) {
		m_hparams = { learningRate, epsilon, beta1, beta2 };
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
    }

    void step() {
        if (m_state.batchAccumulation > 0) {
            step(m_state.batchGradient / m_state.batchAccumulation);
        }

        m_state.batchGradient = 0;
        m_state.batchAccumulation = 0;
    }

    Float variable() const {
        return m_state.variable;
    }

private:
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
        Float epsilon;
        Float beta1;
        Float beta2;
    } m_hparams;
};

struct Sample {
    enum Type {
        TypeBsdf = 0,
        TypeNee,

        TypeCount
    };

    Spectrum estimate;
    Float miWeight;
    Type type;
    std::array<Float, TypeCount> pdfs;
    std::array<Float, TypeCount> betas;

    Spectrum value() const {
        return estimate * pdfs[type];
    }

    Spectrum weightWithoutMIS() const {
        return estimate;
    }

    Spectrum weightWithMIS() const {
        return estimate * miWeight;
    }

    Float mixturePdf() const {
        return mixturePdf(betas);
    }

    Float mixturePdf(const std::array<Float, TypeCount> &betas) const {
        Float sum { 0. };
        for (int type = 0; type < TypeCount; type++)
            sum += betas[type] * pdfs[type];
        return sum;
    }
};

class SamplingFractionOptimizer {
public:
    virtual ~SamplingFractionOptimizer() {}
    virtual SamplingFractionOptimizer *clone() const = 0;
    virtual void append(const Sample &sample) {}
    virtual void step() {}
    virtual void output(StatsRecursiveValues &stats) const {}
    virtual Float bsdfFraction() const = 0;
};

class ConstantOptimizer : public SamplingFractionOptimizer {
private:
    Float value;

public:
    SamplingFractionOptimizer *clone() const override { return &(*new ConstantOptimizer() = *this); }

    ConstantOptimizer() : value(0.5f) {}
    ConstantOptimizer(Float value) : value(value) {}

    Float bsdfFraction() const override {
        return value;
    }
};

class BruteForceOptimizer : public SamplingFractionOptimizer {
private:
    int numCandidates { 101 };
    bool useCrossEstimates { true };
    bool clearOnStep { false };
    std::vector<Sample> samples;

    Float value { 0.5f };
    Spectrum variance { 0. };

    Spectrum estimateVariance(Float bsdfFraction) const {
        const std::array<Float, Sample::TypeCount> betas { bsdfFraction, 1 - bsdfFraction };

        Spectrum fm { 0. };
        Spectrum sm { 0. };

        /**
         * Our goal: Estimating the image variance
         * I = { I_bsdf /      bsdfFraction  with      bsdfFraction  chance
         *     { I_nee  / (1 - bsdfFraction) with (1 - bsdfFraction) chance
         *
         * I_bsdf = g(x) / p_bsdf(x) * w_bsdf(x)
         * w_bsdf(x) = bsdfFraction * p_bsdf(x) / (bsdfFraction * p_bsdf(x) + (1 - bsdfFraction) * p_nee(x))
         *
         * E[I  ] = I_bsdf + I_nee
         * E[I^2] = E[I_bsdf^2] / bsdfFraction + E[I_nee^2] / (1 - bsdfFraction)
         *
         * V[I] = E[I^2] - (I_bsdf + I_nee)^2
         */

        for (const auto &sample : samples) {
            // estimate first moment using actual beta values to reduce noise
            fm += sample.weightWithMIS() / sample.betas[sample.type];
        }

        if (useCrossEstimates) {
            for (const auto &sample : samples) {
                if (sample.estimate.isZero()) continue;

                for (int type = 0; type < Sample::TypeCount; type++) {
                    if (sample.pdfs[type] == 0) continue;

                    const Float miWeight = betas[type] * sample.pdfs[type] / sample.mixturePdf(betas);

                    const Spectrum contrib = sample.estimate * (miWeight / betas[type]);
                    const Float crossPdf = sample.pdfs[sample.type] / sample.pdfs[type];
                    const Float w = sample.miWeight / sample.betas[sample.type];

                    //fm += contrib * w;
                    sm += contrib * contrib * (crossPdf * w) * betas[type];
                }
            }
        } else {
            for (const auto &sample : samples) {
                if (sample.estimate.isZero()) continue;
                const Float miWeight = betas[sample.type] * sample.pdfs[sample.type] / sample.mixturePdf(betas);
                const Spectrum contrib = sample.estimate * (miWeight / betas[sample.type]);
                sm += contrib * contrib * betas[sample.type] / sample.betas[sample.type];
            }
        }

        // normalize by sample count
        fm *= 1. / samples.size();
        sm *= 1. / samples.size();

        const Spectrum var = sm - fm * fm;
        return var;
    }

public:
    SamplingFractionOptimizer *clone() const override { return &(*new BruteForceOptimizer() = *this); }

    void append(const Sample &sample) override {
        samples.push_back(sample);
    }

    void step() override {
        // always test 0.5 first so that this is the default for regions where variance is constant
        // (makes it visually simpler to compare to other optimizers)
        value = 0.5f;
        Float bestV = estimateVariance(value).average();
        if (bestV == 0) return;

        for (int candidate = 0; candidate < numCandidates; candidate++) {
            const Float t = (candidate + 0.5f) / numCandidates;
            const Spectrum v = estimateVariance(t);
            if (v.average() < bestV) {
                bestV = v.average();
                value = t;
                variance = v;
            }
        }

        if (clearOnStep) samples.clear();
    }

    Float bsdfFraction() const override {
        return value;
    }

    void output(StatsRecursiveValues &stats) const override {
        stats.bestVariance.add(variance);
    }
};

class MeyerOptimizer : public SamplingFractionOptimizer {
    struct Estimator {
        /**
         * We estimate:
         * fm = E~p_t[ g  (x) w_t  (x) / p_t  (x) ]
         * sm = E~p_t[ g^2(x) w_t^2(x) / p_t^2(x) ]
         *
         * We can use samples from another technique s:
         * fm = E~p_s[ g  (x) w_t  (x) /  p_s(x)           ]
         * sm = E~p_s[ g^2(x) w_t^2(x) / (p_s(x) * p_t(x)) ]
         */

        int numSamples { 0 };
        Spectrum firstMomentAcc { 0. };
        Spectrum secondMomentAcc { 0. };

        Spectrum firstMoment { 0. };
        Spectrum secondMoment { 0. };
        Spectrum variance { 0. };

        void append(const Spectrum &value, Float crossPdf, Float weight) {
            numSamples += 1;
            if (!value.isZero()) {
                firstMomentAcc += value * weight;
                secondMomentAcc += value * value * (weight * crossPdf);
            }
        }

        void step() {
            if (numSamples < 4) {
                firstMoment = Spectrum { 0.f };
                secondMoment = Spectrum { 0.f };
                variance = Spectrum { 0.f };
                return;
            }

            firstMoment = firstMomentAcc / numSamples;
            secondMoment = secondMomentAcc / numSamples;
            variance = numSamples / Float(numSamples - 1) * (secondMoment - firstMoment * firstMoment);

            numSamples = 0;
            firstMomentAcc = Spectrum { 0. };
            secondMomentAcc = Spectrum { 0. };
        }
    };

    std::array<Estimator, Sample::TypeCount> estimators {};

    Float value { 0.5f };
    bool useCrossEstimates { true };

public:
    SamplingFractionOptimizer *clone() const override { return &(*new MeyerOptimizer() = *this); }

    void append(const Sample &sample) override {
        if (!useCrossEstimates) {
            estimators[sample.type].append(sample.weightWithMIS(), 1, 1);
            return;
        }

        for (int type = Sample::TypeBsdf; type < Sample::TypeCount; type++) {
            const Float miWeight = sample.pdfs[type] * sample.betas[type] / sample.mixturePdf();

            estimators[type].append(
                sample.weightWithoutMIS() * miWeight,
                sample.pdfs[sample.type] / sample.pdfs[type],
                sample.miWeight / sample.betas[sample.type]
            );
        }
    }

    void step() override {
        for (auto &estimator : estimators) estimator.step();

        const Float a = std::sqrt(estimators[Sample::TypeBsdf].secondMoment.average());
        const Float b = std::sqrt(estimators[Sample::TypeNee ].secondMoment.average());
        if (a + b == 0) return;

        value = a / (a + b);
        value = math::clamp(value, Float(0.01f), Float(0.99f));
    }

    void output(StatsRecursiveValues &stats) const override {
        stats.neeFirstMoment.add(estimators[Sample::TypeNee].firstMoment);
        stats.neeSecondMoment.add(estimators[Sample::TypeNee].secondMoment);
        stats.neeVariance.add(estimators[Sample::TypeNee].variance);

        stats.bsdfFirstMoment.add(estimators[Sample::TypeBsdf].firstMoment);
        stats.bsdfSecondMoment.add(estimators[Sample::TypeBsdf].secondMoment);
        stats.bsdfVariance.add(estimators[Sample::TypeBsdf].variance);
    }

    Float bsdfFraction() const override {
        return value;
    }
};

class LuOptimizer : public SamplingFractionOptimizer {
    Float value { 0.5f };

    struct Statistics {
        Float term1 { 0. };
        Float term2 { 0. };
    } statistics;

public:
    SamplingFractionOptimizer *clone() const override { return &(*new LuOptimizer() = *this); }

    void append(const Sample &sample) override {
        if (sample.estimate.isZero()) return;

        const Float mixtureP = (sample.pdfs[Sample::TypeBsdf] + sample.pdfs[Sample::TypeNee]) / 2;
        const Float deltaP   = (sample.pdfs[Sample::TypeBsdf] - sample.pdfs[Sample::TypeNee]) / 2;

        const Spectrum c = sample.value() * (1 / mixtureP);
        const Float base = (c * c).average() * sample.miWeight / (sample.betas[sample.type] * sample.pdfs[sample.type]);
        statistics.term1 += base * deltaP;
        statistics.term2 += base * deltaP * deltaP / mixtureP;
    }

    void step() override {
        if (statistics.term2 != 0) {
            value = 0.25f * (2 + statistics.term1 / statistics.term2);
            value = math::clamp(value, Float(0.025f), Float(0.975f));
        }
        statistics = {};
    }

    Float bsdfFraction() const override {
        return value;
    }
};

class SzirmayOptimizer : public SamplingFractionOptimizer {
    Float value { 0.5f };

    struct Statistics {
        Float C { 0. };
        Float dCda { 0. };
    } statistics;

public:
    SamplingFractionOptimizer *clone() const override { return &(*new SzirmayOptimizer() = *this); }

    void append(const Sample &sample) override {
        if (sample.estimate.isZero()) return;

        const Float mixtureP = sample.mixturePdf();
        const Float deltaP = sample.pdfs[Sample::TypeBsdf] - sample.pdfs[Sample::TypeNee];
        statistics.C += sample.value().average() * deltaP / (mixtureP * mixtureP);
        statistics.dCda += -2 * sample.value().average() * deltaP * deltaP / (mixtureP * mixtureP * mixtureP);
    }

    void step() override {
        if (statistics.dCda != 0) {
            value = value - statistics.C / statistics.dCda;
            value = math::clamp(value, Float(0.025f), Float(0.975f));
        }
        statistics = {};
    }

    Float bsdfFraction() const override {
        return value;
    }
};

class SbertOptimizer : public SamplingFractionOptimizer {
    Float value { 0.5f };

    struct Statistics {
        Float dVda1 { 0. };
        Float dVda2 { 0. };
    } statistics;

public:
    SamplingFractionOptimizer *clone() const override { return &(*new SbertOptimizer() = *this); }

    void append(const Sample &sample) override {
        if (sample.estimate.isZero()) return;

        const Float mixtureP = sample.mixturePdf();
        const Float deltaP = sample.pdfs[Sample::TypeNee] - sample.pdfs[Sample::TypeBsdf];
        const Float f2 = (sample.value() * sample.value()).average();

        statistics.dVda1 += f2 * deltaP / ((mixtureP * mixtureP) * mixtureP);
        statistics.dVda2 += f2 * (deltaP * deltaP) / ((mixtureP * mixtureP) * (mixtureP * mixtureP));
    }

    void step() override {
        if (statistics.dVda2 != 0) {
            value = value - statistics.dVda1 / statistics.dVda2;
            value = math::clamp(value, Float(0.025f), Float(0.975f));
        }
        statistics = {};
    }

    Float bsdfFraction() const override {
        return value;
    }
};

class MullerOptimizer : public SamplingFractionOptimizer {
    Float ratioPower { 1.f };
    AdamOptimizer adamOptimizer { 0.1f };

    inline Float logistic(Float x) const {
        return 1 / (1 + std::exp(-x));
    }

    inline Float bsdfSamplingFraction(Float variable) const {
        return logistic(variable);
    }

    inline Float dBsdfSamplingFraction_dVariable(Float variable) const {
        Float fraction = bsdfSamplingFraction(variable);
        return fraction * (1 - fraction);
    }

public:
    SamplingFractionOptimizer *clone() const override { return &(*new MullerOptimizer() = *this); }

    MullerOptimizer() {}
    MullerOptimizer(Float ratioPower) : ratioPower(ratioPower) {}

    void append(const Sample &sample) override {
        Float contribution = sample.value().average();

        // GRADIENT COMPUTATION
        Float variable = adamOptimizer.variable();
        Float samplingFraction = bsdfSamplingFraction(variable);

        // Loss gradient w.r.t. sampling fraction
        Float mixPdf = samplingFraction * sample.pdfs[Sample::TypeBsdf] + (1 - samplingFraction) * sample.pdfs[Sample::TypeNee];
        Float ratio = std::pow(contribution / mixPdf, ratioPower);
        Float dLoss_dSamplingFraction = -ratio / mixPdf * (sample.pdfs[Sample::TypeBsdf] - sample.pdfs[Sample::TypeNee]);

        // Chain rule to get loss gradient w.r.t. trainable variable
        Float dLoss_dVariable = dLoss_dSamplingFraction * dBsdfSamplingFraction_dVariable(variable);

        // We want some regularization such that our parameter does not become too big.
        // We use l2 regularization, resulting in the following linear gradient.
        Float l2RegGradient = 0.01f * variable;

        Float lossGradient = l2RegGradient + dLoss_dVariable;

        // ADAM GRADIENT DESCENT
        adamOptimizer.append(lossGradient, 1.f);
    }

    void step() override {
        adamOptimizer.step();
    }

    Float bsdfFraction() const override {
        return bsdfSamplingFraction(adamOptimizer.variable());
    }
};

class MISampleAllocationPathTracer : public MonteCarloIntegrator {
public:
    MISampleAllocationPathTracer(const Properties &props)
    : MonteCarloIntegrator(props) {
        Log(EInfo, "running commit %s", g_GIT_SHA1);
        for (const auto &name : props.getPropertyNames()) {
            Log(EInfo, "%s: %s", name.c_str(), props.getAsString(name).c_str());
        }

        auto optimizerName = props.getString("optimizer", "constant");
        if (optimizerName == "constant")
            m_optimizer.reset(new ConstantOptimizer(props.getFloat("bsdfFraction", 0.5f)));
        else if (optimizerName == "bruteforce")
            m_optimizer.reset(new BruteForceOptimizer());
        else if (optimizerName == "meyer")
            m_optimizer.reset(new MeyerOptimizer());
        else if (optimizerName == "muller")
            m_optimizer.reset(new MullerOptimizer(props.getFloat("mullerPower", 1.f)));
        else if (optimizerName == "lu")
            m_optimizer.reset(new LuOptimizer());
        else if (optimizerName == "szirmay")
            m_optimizer.reset(new SzirmayOptimizer());
        else if (optimizerName == "sbert")
            m_optimizer.reset(new SbertOptimizer());
    }

    inline bool exists (const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int sensorResID,
                    int samplerResID) {
        /* Check if output path exists, otherwise create the requested directory */
        if (!exists(scene->getDestinationFile().parent_path().string())) {
            if (mkdir(scene->getDestinationFile().parent_path().string().c_str(), 0775) == -1)
                return false;
        }
        return true;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {

#ifdef EARS_INCLUDE_AOVS
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

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
#endif

        auto result = SamplingIntegrator::render(scene, queue, job, sceneResID, sensorResID, samplerResID);

#ifdef EARS_INCLUDE_AOVS
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
            std::string suffix = "-dbg";
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

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

#ifdef EARS_INCLUDE_AOVS
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

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            //if (stop)
            //    break;

            Float choiceRnd { rRec.nextSample1D() };

            std::unique_ptr<SamplingFractionOptimizer> optimizer { m_optimizer->clone() };

            const int samplesPerIter = 128;//32;
            const int iterCount = sampler->getSampleCount() / samplesPerIter;
            for (int iter = 0; iter < iterCount; iter++) {
                const bool isLastIter = iter == (iterCount - 1);
                for (int j = 0; j < samplesPerIter; j++) {
                    stats.reset();

                    rRec.newQuery(queryType, sensor->getMedium());
                    Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                    if (needsApertureSample)
                        apertureSample = rRec.nextSample2D();
                    if (needsTimeSample)
                        timeSample = rRec.nextSample1D();

                    Spectrum spec = sensor->sampleRayDifferential(
                        sensorRay, samplePos, apertureSample, timeSample);

                    Spectrum contrib = spec * Li(sensorRay, rRec, *optimizer, stats, choiceRnd);
                    block->put(samplePos, contrib, rRec.alpha);
                    sampler->advance();

#ifdef EARS_INCLUDE_AOVS
                    if (isLastIter) {
                        stats.fraction.add(optimizer->bsdfFraction());
                        stats.pixel2nd.add(contrib * contrib);
                        optimizer->output(stats);
                        stats.put(blocks, samplePos, rRec.alpha);
                    }
#endif
                }

                optimizer->step();
            }
        }

#ifdef EARS_INCLUDE_AOVS
        m_statsImages->put(blocks);
#endif
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Assert(false);
        return Spectrum { 0.f };
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec, SamplingFractionOptimizer &optimizer, StatsRecursiveValues &stats, Float &choiceRnd) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);

        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        if (!its.isValid()) {
            return scene->evalEnvironment(ray);
        }

        Spectrum Li(0.0f);

        if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && !m_hideEmitters)
            Li += its.Le(-ray.d);

        const BSDF *bsdf = its.getBSDF(ray);

        // decide which technique to use
        const Float bsdfFraction = optimizer.bsdfFraction();
        bool useBsdfEstimator = false;
        //if ((choiceRnd += bsdfFraction) >= 1) {
        //    choiceRnd -= 1;
        //    useBsdfEstimator = true;
        //}
        useBsdfEstimator = rRec.nextSample1D() < bsdfFraction;

        DirectSamplingRecord dRec(its);
        if (useBsdfEstimator) {
            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);

            bool hitEmitter = false;
            Spectrum value;

            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                const Emitter *env = scene->getEnvironmentEmitter();
                if (env) {
                    value = env->evalEnvironment(ray);
                    if (env->fillDirectSamplingRecord(dRec, ray)) {
                        hitEmitter = true;
                    }
                }
            }

            if (hitEmitter) {
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;
                const Float weight = miWeight(bsdfFraction * bsdfPdf, (1 - bsdfFraction) * lumPdf);
                Li += value * bsdfWeight * weight / bsdfFraction;
                stats.liBsdf.add(value * bsdfWeight / bsdfFraction);
                stats.misBsdf.add(value * bsdfWeight * weight / bsdfFraction);

                optimizer.append({
                    .estimate = value * bsdfWeight,
                    .miWeight = weight,
                    .type = Sample::TypeBsdf,
                    .pdfs = { bsdfPdf, lumPdf },
                    .betas = { bsdfFraction, 1 - bsdfFraction }
                });
            } else {
                stats.liBsdf.add(Spectrum(0.0f));
                stats.misBsdf.add(Spectrum(0.0f));

                optimizer.append({
                    .estimate = Spectrum { 0. },
                    .miWeight = 0,
                    .type = Sample::TypeBsdf,
                    .pdfs = { bsdfPdf, 0 },
                    .betas = { bsdfFraction, 1 - bsdfFraction }
                });
            }
        } else {
            /* Estimate the direct illumination if this is requested */
            if ((bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                const Spectrum bsdfVal = bsdf->eval(bRec);
                Float bsdfPdf = bsdf->pdf(bRec);

                Float weight = miWeight((1 - bsdfFraction) * dRec.pdf, bsdfFraction * bsdfPdf);
                Li += value * bsdfVal * weight / (1 - bsdfFraction);
                stats.liNee.add(value * bsdfVal / (1 - bsdfFraction));
                stats.misNee.add(value * bsdfVal * weight / (1 - bsdfFraction));

                optimizer.append({
                    .estimate = value * bsdfVal,
                    .miWeight = weight,
                    .type = Sample::TypeNee,
                    .pdfs = { bsdfPdf, dRec.pdf },
                    .betas = { bsdfFraction, 1 - bsdfFraction }
                });
            } else {
                stats.liNee.add(Spectrum(0.f));
                stats.misNee.add(Spectrum(0.f));

                optimizer.append({
                    .estimate = Spectrum { 0. },
                    .miWeight = 0,
                    .type = Sample::TypeNee,
                    .pdfs = { 0, dRec.pdf },
                    .betas = { bsdfFraction, 1 - bsdfFraction }
                });
            }
        }

        return Li;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        //pdfA *= pdfA;
        //pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "MISampleAllocationPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

private:
    std::unique_ptr<SamplingFractionOptimizer> m_optimizer;

    std::unique_ptr<StatsRecursiveImageBlocks> m_statsImages;
    mutable ref<ImageBlock> m_debugImage;
    mutable ref<Film> m_debugFilm;

    mutable std::unique_ptr<ProgressReporter> m_progress;
    std::chrono::steady_clock::time_point m_startTime;

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(MISampleAllocationPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MISampleAllocationPathTracer, "MI sample allocation path tracer");
MTS_NAMESPACE_END
