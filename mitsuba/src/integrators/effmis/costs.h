#pragma once

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

struct EffMISCosts
{
    /// Number of pixels equals number of camera paths
    int NumPixels; 

    /// Average length of camera paths
    Float AverageCameraPathLength;

    /// Average length of light paths
    Float AverageLightPathLength;

    void update(int numPixels, Float avgCamLen, Float avgLightLen);

    Float evaluateForPixel(Float numConnections) const;
    Float evaluateGlobal(Float numConnections) const;
};

MTS_NAMESPACE_END