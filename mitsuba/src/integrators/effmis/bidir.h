#pragma once

#include <mitsuba/bidir/common.h>
#include <mitsuba/bidir/path.h>

MTS_NAMESPACE_BEGIN

constexpr int ProxyNumConnections = 4;

struct BidirPdfs
{
    Float *PdfRad; // PdfRad
    Float *PdfImp;
    bool* Connectable;
    bool* IsNull;
    const size_t MaxLength;
    int EmitterRefIndirection; // Reference s for NEE
    int SensorRefIndirection;  // Reference t for LT
    Float RatioEmitterDirect;
    Float RatioSensorDirect;

    static BidirPdfs create(size_t length, Float *pdfRad, Float *pdfImp, bool* connectable, bool* isNull)
    {
        return BidirPdfs{pdfRad, pdfImp, connectable, isNull, length, 0, 0, 0, 0};
    }

    void gather(Scene* scene, const Path& emitterSubpath, const PathEdge* connectionEdge, const Path& sensorSubpath, int s, int t,
        bool useNEE, bool useLT) {
        int k = s+t+1, n = k+1;

        SAssert(n <= (int)MaxLength);
        
        RatioEmitterDirect = 0;
        RatioSensorDirect = 0;

        const PathVertex
            *vsPred = emitterSubpath.vertexOrNull(s-1),
            *vtPred = sensorSubpath.vertexOrNull(t-1),
            *vs = emitterSubpath.vertex(s),
            *vt = sensorSubpath.vertex(t);

        /* Keep track of which vertices are connectable / null interactions */
        int pos = 0;
        for (int i=0; i<=s; ++i) {
            const PathVertex *v = emitterSubpath.vertex(i);
            Connectable[pos] = v->isConnectable();
            IsNull[pos] = v->isNullInteraction() && !Connectable[pos];
            pos++;
        }

        for (int i=t; i>=0; --i) {
            const PathVertex *v = sensorSubpath.vertex(i);
            Connectable[pos] = v->isConnectable();
            IsNull[pos] = v->isNullInteraction() && !Connectable[pos];
            pos++;
        }

        if (k <= 3)
            useNEE = false;

        EMeasure vsMeasure = EArea, vtMeasure = EArea;
        if (useNEE || useLT) {
            /* When direct sampling is enabled, we may be able to create certain
            connections that otherwise would have failed (e.g. to an
            orthographic camera or a directional light source) */
            const AbstractEmitter *emitter = (s > 0 ? emitterSubpath.vertex(1) : vt)->getAbstractEmitter();
            const AbstractEmitter *sensor = (t > 0 ? sensorSubpath.vertex(1) : vs)->getAbstractEmitter();

            EMeasure emitterDirectMeasure = emitter->getDirectMeasure();
            EMeasure sensorDirectMeasure  = sensor->getDirectMeasure();

            if (useNEE) {
                Connectable[0]   = emitterDirectMeasure != EDiscrete && emitterDirectMeasure != EInvalidMeasure;
                Connectable[1]   = emitterDirectMeasure != EInvalidMeasure;
            }
            
            if (useLT) {
                Connectable[k-1] = sensorDirectMeasure != EInvalidMeasure;
                Connectable[k]   = sensorDirectMeasure != EDiscrete && sensorDirectMeasure != EInvalidMeasure;
            }

            /* The following is needed to handle orthographic cameras &
            directional light sources together with direct sampling */
            if (useLT && t == 1)
                vtMeasure = sensor->needsDirectionSample() ? EArea : EDiscrete;
            else if (useNEE && s == 1)
                vsMeasure = emitter->needsDirectionSample() ? EArea : EDiscrete;
        }

        /* Collect importance transfer area/volume densities from vertices */
        pos = 0;
        PdfImp[pos++] = 1.0;

        for (int i=0; i<s; ++i)
            PdfImp[pos++] = emitterSubpath.vertex(i)->pdf[EImportance]
                * emitterSubpath.edge(i)->pdf[EImportance];

        PdfImp[pos++] = vs->evalPdf(scene, vsPred, vt, EImportance, vsMeasure)
            * connectionEdge->pdf[EImportance];

        if (t > 0) {
            PdfImp[pos++] = vt->evalPdf(scene, vs, vtPred, EImportance, vtMeasure)
                * sensorSubpath.edge(t-1)->pdf[EImportance];

            for (int i=t-1; i>0; --i)
                PdfImp[pos++] = sensorSubpath.vertex(i)->pdf[EImportance]
                    * sensorSubpath.edge(i-1)->pdf[EImportance];
        }

        /* Collect radiance transfer area/volume densities from vertices */
        pos = 0;
        if (s > 0) {
            for (int i=0; i<s-1; ++i)
                PdfRad[pos++] = emitterSubpath.vertex(i+1)->pdf[ERadiance]
                    * emitterSubpath.edge(i)->pdf[ERadiance];

            PdfRad[pos++] = vs->evalPdf(scene, vt, vsPred, ERadiance, vsMeasure)
                * emitterSubpath.edge(s-1)->pdf[ERadiance];
        }

        PdfRad[pos++] = vt->evalPdf(scene, vtPred, vs, ERadiance, vtMeasure)
            * connectionEdge->pdf[ERadiance];

        for (int i=t; i>0; --i)
            PdfRad[pos++] = sensorSubpath.vertex(i-1)->pdf[ERadiance]
                * sensorSubpath.edge(i-1)->pdf[ERadiance];

        PdfRad[pos++] = 1.0;

        /* When the path contains specular surface interactions, it is possible
        to compute the correct MI weights even without going through all the
        trouble of computing the proper generalized geometric terms (described
        in the SIGGRAPH 2012 specular manifolds paper). The reason is that these
        all cancel out. But to make sure that that's actually true, we need to
        convert some of the area densities in the 'PdfRad' and 'PdfImp' arrays
        into the projected solid angle measure */
        for (int i=1; i <= k-3; ++i) {
            if (i == s || !(Connectable[i] && !Connectable[i+1]))
                continue;

            const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k-i);
            const PathVertex *succ = i+1 <= s ? emitterSubpath.vertex(i+1) : sensorSubpath.vertex(k-i-1);
            const PathEdge *edge = i < s ? emitterSubpath.edge(i) : sensorSubpath.edge(k-i-1);

            PdfImp[i+1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface()  ? dot(edge->d, cur->getGeometricNormal())  : 1));
        }

        for (int i=k-1; i >= 3; --i) {
            if (i-1 == s || !(Connectable[i] && !Connectable[i-1]))
                continue;

            const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k-i);
            const PathVertex *succ = i-1 <= s ? emitterSubpath.vertex(i-1) : sensorSubpath.vertex(k-i+1);
            const PathEdge *edge = i <= s ? emitterSubpath.edge(i-1) : sensorSubpath.edge(k-i);

            PdfRad[i-1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface()  ? dot(edge->d, cur->getGeometricNormal())  : 1));
        }

        EmitterRefIndirection = 2, SensorRefIndirection = k-2;

        /* One more array sweep before the actual useful work starts -- phew! :)
        "Collapse" edges/vertices that were caused by BSDF::ENull interactions.
        The BDPT implementation is smart enough to connect straight through those,
        so they shouldn't be treated as Dirac delta events in what follows */
        for (int i=1; i <= k-3; ++i) {
            if (!Connectable[i] || !IsNull[i+1])
                continue;

            int start = i+1, end = start;
            while (IsNull[end+1])
                ++end;

            if (!Connectable[end+1]) {
                /// The chain contains a non-ENull interaction
                IsNull[start] = false;
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

            PdfRad[start-1] *= PdfRad[end] * geoTerm;
            PdfRad[end] = 1;
            PdfImp[start] *= PdfImp[end+1] * geoTerm;
            PdfImp[end+1] = 1;

            /* When an ENull chain starts right after the emitter / before the sensor,
            we must keep track of the reference vertex for direct sampling strategies. */
            if (start == 2)
                EmitterRefIndirection = end + 1;
            else if (end == k-2)
                SensorRefIndirection = start - 1;

            i = end;
        }

        /* When direct sampling strategies are enabled, we must
        account for them here as well */
        if (useNEE) {
            /* Direct connection probability of the emitter */
            const PathVertex *sample = s>0 ? emitterSubpath.vertex(1) : vt;
            const PathVertex *ref = EmitterRefIndirection <= s
                ? emitterSubpath.vertex(EmitterRefIndirection) : sensorSubpath.vertex(k-EmitterRefIndirection);
            EMeasure measure = sample->getAbstractEmitter()->getDirectMeasure();

            if (Connectable[1] && Connectable[EmitterRefIndirection])
                RatioEmitterDirect = ref->evalPdfDirect(scene, sample, EImportance,
                    measure == ESolidAngle ? EArea : measure) / PdfImp[1];
        }

        if (useLT) {
            /* Direct connection probability of the sensor */
            const PathVertex *sample = t > 0 ? sensorSubpath.vertex(1) : vs;
            const PathVertex *ref = SensorRefIndirection <= s ? emitterSubpath.vertex(SensorRefIndirection)
                : sensorSubpath.vertex(k-SensorRefIndirection);
            EMeasure measure = sample->getAbstractEmitter()->getDirectMeasure();

            if (Connectable[k-1] && Connectable[SensorRefIndirection])
                RatioSensorDirect = ref->evalPdfDirect(scene, sample, ERadiance,
                    measure == ESolidAngle ? EArea : measure) / PdfRad[k-1];
        }
    }

    Float computeMIS(int s, int t, bool sampleDirect, bool lightImage, Float connectionDensity) const {
        // FIXME: Disabling sampleDirect might introduce bias. Have to investigate (or never disable it)
        if (connectionDensity == 0)
            lightImage = false;
    
        const int k = s+t+1;
        const int minT = lightImage ? 1 : 2;

        SAssert(t >= minT); // No support for direct sensor stuff

        if (k <= 3)
            sampleDirect = false;

        const int bidirStart = sampleDirect ? 2 : 1;

        double initial = 1.0f;

        /* When direct sampling strategies are enabled, we must
        account for them here as well */
        if (sampleDirect && s == 1) {      // NEE
            if (RatioEmitterDirect == 0)
                return 0;
            SAssert(RatioEmitterDirect > 0);
            initial /= RatioEmitterDirect;
        } else if (t == 1) {               // LT
            SAssert(lightImage);
            if (RatioSensorDirect == 0)
                return 0;
            SAssert(RatioSensorDirect > 0);
            initial /= RatioSensorDirect;
        }
        else if (s >= bidirStart && t > 1) // VC
            initial /= connectionDensity;

        SAssert(std::isfinite(initial));
        double sumReciprocal = 1, pdf = initial;

        /* With all of the above information, the MI weight can now be computed.
        Since the goal is to evaluate the balance heuristic, the absolute area
        product density of each strategy is interestingly not required. Instead,
        an incremental scheme can be used that only finds the densities relative
        to the (s,t) strategy, which can be done using a linear sweep. For
        details, refer to the Veach thesis, p.306. */
        for (int i=s+1; i<k; ++i) {
            double next = pdf * (double) PdfImp[i] / (double) PdfRad[i],
                value = next;

            SAssert(std::isfinite(next));

            const int tPrime = k-i-1;

            if (sampleDirect && i == 1)             // NEE
                value *= RatioEmitterDirect;
            else if (lightImage && tPrime == 1)     // LT
                value *= RatioSensorDirect;
            else if (i >= bidirStart && tPrime > 1) // VC
                value *= connectionDensity;

            if (Connectable[i] && (Connectable[i+1] || IsNull[i+1]) && tPrime >= minT)
                sumReciprocal += value; // Balance heuristic

            pdf = next;
        }

        /* As above, but now compute pdf[i] with i<s (this is done by
        evaluating the inverse of the previous expressions). */
        pdf = initial;
        for (int i=s-1; i>=0; --i) {
            double next = pdf * (double) PdfRad[i+1] / (double) PdfImp[i+1],
                  value = next;

            SAssert(std::isfinite(next));

            const int tPrime = k-i-1;

            if (sampleDirect && i == 1)             // NEE
                value *= RatioEmitterDirect;
            else if (lightImage && tPrime == 1)     // LT
                value *= RatioSensorDirect;
            else if (i >= bidirStart && tPrime > 1) // VC
                value *= connectionDensity;

            if (Connectable[i] && (Connectable[i+1] || IsNull[i+1]) && tPrime >= minT)
                sumReciprocal += value; // Balance heuristic

            pdf = next;
        }

        SAssert(std::isfinite(sumReciprocal));
        SAssert(sumReciprocal >= 1);
        return 1 / sumReciprocal;
    }

    struct ProxyWeights
    {
        Float PathTracing; // Combined value of emission hits and NEE
        Float Connection;
    };

    inline ProxyWeights computeProxyWeights(int k, bool sampleDirect, bool lightImage) const
    {
        SAssert(k > 2);

        // t = k - 1 - s
        const Float misHit = computeMIS(0, k-1, sampleDirect, lightImage, ProxyNumConnections);
        const Float misNEE = (sampleDirect && k > 3) ? computeMIS(1, k-2, sampleDirect, lightImage, ProxyNumConnections) : 0;
        const Float misPT  = misHit + misNEE;

        // MIS weights must sum to one. We don't compute the connection weights explicitly, so make sure
        // that everything else is below one, up to some numerical error margin.
        SAssert(misPT <= (Float)1.0001);
        SAssert(misPT >= 0);

        return ProxyWeights{
            misPT,
            std::min<Float>(std::max<Float>(1 - misPT, 0), 1), // Prevent negative moments due to small numerical errors
        };
    }
};

class BidirUtils {
public:
    static inline void computeCachedWeights(const Path& subpath, Spectrum* weights, ETransportMode mode)
    {
        weights[0] = Spectrum(1.0f);
        for (size_t i = 1; i < subpath.vertexCount(); ++i)
            weights[i] = weights[i - 1]
                        * subpath.vertex(i - 1)->weight[mode] 
                        * subpath.vertex(i - 1)->rrWeight 
                        * subpath.edge(i - 1)->weight[mode];
    }
};

MTS_NAMESPACE_END