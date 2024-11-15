#include "costs.h"

MTS_NAMESPACE_BEGIN

// Computed via time measurements
constexpr Float CostTrace   = 1.8f; // Normalized cost for tracing a vertex (camera or light path)
constexpr Float CostNEE     = 1.0f; // Normalized cost for NEE
constexpr Float CostConnect = 2.0f; // Normalized cost for connection

void EffMISCosts::update(int numPixels, Float avgCamLen, Float avgLightLen)
{
    NumPixels = numPixels;

    constexpr Float DefaultAverageCamLength = 3.0f;

    AverageCameraPathLength = avgCamLen <= 0.1f ? DefaultAverageCamLength : avgCamLen;
    AverageLightPathLength = avgLightLen <= 0.1f ? AverageCameraPathLength : avgLightLen;
}

Float EffMISCosts::evaluateForPixel(Float numConnections) const
{
    const Float neeTime = AverageCameraPathLength * CostNEE;
    const Float connectTime = AverageCameraPathLength * AverageLightPathLength * numConnections * CostConnect;
    const Float traceTime = (AverageCameraPathLength + AverageLightPathLength * numConnections) * CostTrace;

    const Float result = neeTime + connectTime + traceTime;
    return result;
}

Float EffMISCosts::evaluateGlobal(Float numConnections) const
{
    return evaluateForPixel(numConnections); // No difference ;)
}

MTS_NAMESPACE_END