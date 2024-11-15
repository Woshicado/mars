#include "optimizer.h"
#include "effmis.h"
#include "effmis_wr.h"

#include <vector>

MTS_NAMESPACE_BEGIN

/// 3x3 kernel with symmetric function and clamp border
template<typename Functor>
static inline void applyFilter(Bitmap* bitmap, Functor func) {
    SAssert(bitmap->getChannelCount() == 1);
    SAssert(bitmap->getComponentFormat() == Bitmap::EFloat);

    ref<Bitmap> clone = bitmap->clone();

    const auto size = bitmap->getSize();
    const int slice = size.x;

    SAssert(size.x >= 3 && size.y >= 3);
    
#define _INDEX(y, x) ((y)*slice+(x))

    // Horizontal sweep
    Float* dst = clone->getFloatData();
    const Float* src = bitmap->getFloatData();
#if defined(MTS_OPENMP)
    #pragma omp parallel for
#endif
    for (int y = 0; y < size.y; ++y) {
        dst[_INDEX(y, 0)] = func(src[_INDEX(y, 0)], src[_INDEX(y, 0)], src[_INDEX(y, 1)]);
        for (int x = 1; x < size.x-1; ++x)
            dst[_INDEX(y, x)] = func(src[_INDEX(y, x)], src[_INDEX(y, x-1)], src[_INDEX(y, x+1)]);
        dst[_INDEX(y, size.x-1)] = func(src[_INDEX(y, size.x-2)], src[_INDEX(y, size.x-1)], src[_INDEX(y, size.x-1)]);
    }

    // Vertical sweep
    dst = bitmap->getFloatData();
    src = clone->getFloatData();
#if defined(MTS_OPENMP)
    #pragma omp parallel for
#endif
    for (int x = 0; x < size.x; ++x) {
        dst[_INDEX(0, x)] = func(src[_INDEX(0, x)],src[_INDEX(0, x)], src[_INDEX(1, x)]);
        for (int y = 1; y < size.y-1; ++y)
            dst[_INDEX(y, x)] = func(src[_INDEX(y, x)], src[_INDEX(y-1, x)], src[_INDEX(y+1, x)]);
        dst[_INDEX(size.y-1, x)] = func(src[_INDEX(size.y-2, x)], src[_INDEX(size.y-1, x)], src[_INDEX(size.y-1, x)]);
    }

#undef _INDEX
}

static inline Float filter_max(Float a, Float b, Float c) {
    return std::max(a, std::max(b, c));
}
static inline Float filter_min(Float a, Float b, Float c) {
    return std::min(a, std::min(b, c));
}
static inline Float filter_box(Float a, Float b, Float c) {
    return (a + b + c) / 3;
}

static inline void dilate(Bitmap* bitmap) {
    applyFilter(bitmap, filter_max);
}
static inline void erode(Bitmap* bitmap) {
    applyFilter(bitmap, filter_min);
}
static inline void box(Bitmap* bitmap) {
    applyFilter(bitmap, filter_box);
}

void EffMISOptimizer::setup(const EffMISContext& context, EffMISWorkResult* wr) 
{
    SAssert(!context.connectionCandidates.empty());
 
#if 0
    // This lets some scenes only use PT in the end, but it is way more robust.

    // Same as in AdaptiveVCM::FilterMoments() (Paper)
    for(int i = 0; i < 3; ++i) {
        for (size_t index = 0; index < context.connectionCandidates.size(); ++index)
            box(wr->getMoments(index)->getBitmap());
    }
#endif
}

int EffMISOptimizer::optimizeGlobal(const EffMISContext& context, const EffMISWorkResult* wr, const EffMISCosts& costHeuristic, Bitmap* pixelEstimate) 
{
    SAssert(!context.connectionCandidates.empty());
    
    Float* moments = (Float*)alloca(context.connectionCandidates.size() * sizeof(Float));
    Float* costs = (Float*)alloca(context.connectionCandidates.size() * sizeof(Float));

    const Float recipNumPixels = 1 / (Float)context.numberOfPixels();

    if (pixelEstimate) { // RelMSE
        const auto size = context.cropSize;

#if defined(MTS_OPENMP)
        #pragma omp parallel for
#endif
        for (size_t index = 0; index < context.connectionCandidates.size(); ++index) {
            const int c = context.connectionCandidates[index];

            const Bitmap* moment_bitmap = wr->getMoments(index)->getBitmap();
            const Float cost = costHeuristic.evaluateGlobal(c);

            for (int y = 0; y < size.y; ++y) {
                for (int x = 0; x < size.x; ++x) {
                    const Point2i pixel = Point2i(x, y);

                    const Float mean = pixelEstimate->getPixel(pixel).average();
                    if (mean == 0) // Skip black pixels
                        continue;
                    const Float recipMeanSquare = 1.0f / (mean * mean + 0.0001f);

                    const Float moment = moment_bitmap->getPixel(pixel).average();
                    moments[index] += moment * recipMeanSquare * recipNumPixels;
                    costs[index] += cost * recipNumPixels;
                }
            }
        }
    } else {
#if defined(MTS_OPENMP)
        #pragma omp parallel for
#endif
        for (size_t index = 0; index < context.connectionCandidates.size(); ++index) {
            const int c = context.connectionCandidates[index];

            moments[index] = wr->getMoments(index)->getBitmap()->average().average();
            costs[index] = costHeuristic.evaluateGlobal(c) * recipNumPixels;
        }
    }

    // Pick best candidate
    Float bestWorkNorm = std::numeric_limits<Float>::max();
    int bestCandidate = context.connectionCandidates[0];

    size_t index = 0;
    for (const auto& candidate: context.connectionCandidates) {
        const Float cost = costs[index];
        const Float moment = moments[index];
        const Float workNorm = moment * cost;

        if (workNorm < bestWorkNorm)
        {
            bestWorkNorm = workNorm;
            bestCandidate = candidate;
        }
        ++index;
    }

    return bestCandidate;
}

int EffMISOptimizer::optimizePerPixel(EffMISContext& context, const EffMISWorkResult* wr, const EffMISCosts& costHeuristic, Bitmap* pixelEstimate) 
{
    SAssert(!context.connectionCandidates.empty());
    SAssert(context.numConnectionsImage);

    const auto size = context.cropSize; 

#if defined(MTS_OPENMP)
    #pragma omp parallel for
#endif
    for (int y = 0; y < size.y; ++y) {
        Float* data = context.numConnectionsImage->getFloatData() + y * size.x;
        for (int x = 0; x < size.x; ++x) {
            const Point2i pixel = Point2i(x, y);

            // Pick best candidate
            Float bestWorkNorm = std::numeric_limits<Float>::max();
            int bestC = context.connectionCandidates[0];
            size_t index = 0;
            for (const int c : context.connectionCandidates) {
                const Float cost = costHeuristic.evaluateForPixel(c);
                const Float moment = wr->getMoments(index)->getBitmap()->getPixel(pixel).average();
                const Float workNorm = moment * cost;

                if (workNorm < bestWorkNorm)
                {
                    bestWorkNorm = workNorm;
                    bestC = c;
                }
                ++index;
            }

            *(data++) = (Float)bestC;
        }
    }

    // Apply filters
    // Same as in AdaptiveVCM::FilterConnectMask() (Paper)
    for(int i = 0; i < 3; ++i)
        dilate(context.numConnectionsImage.get());
    // for(int i = 0; i < 3; ++i)
    //     erode(context.numConnectionsImage.get());
    for (int i = 0; i < 16; ++i)
        box(context.numConnectionsImage.get());
    
    // Compute maximum
    int maximum = 0;
    for (int y = 0; y < size.y; ++y) {
        const Float* data = context.numConnectionsImage->getFloatData() + y * size.x;
        for (int x = 0; x < size.x; ++x) {
            maximum = std::max((int)std::ceil(*data++), maximum);
        }
    }
    return maximum;
}

MTS_NAMESPACE_END