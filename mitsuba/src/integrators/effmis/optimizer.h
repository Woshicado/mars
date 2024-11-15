#pragma once

#include "costs.h"

MTS_NAMESPACE_BEGIN

struct EffMISContext;
class EffMISWorkResult;

struct EffMISOptimizer
{
    static void setup(const EffMISContext& config, EffMISWorkResult* wr);
    static int optimizeGlobal(const EffMISContext& config, const EffMISWorkResult* wr, const EffMISCosts& costs, Bitmap* pixelEstimate);
    static int optimizePerPixel(EffMISContext& config, const EffMISWorkResult* wr, const EffMISCosts& costs, Bitmap* pixelEstimate);
};

MTS_NAMESPACE_END