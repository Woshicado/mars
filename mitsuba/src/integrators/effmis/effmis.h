/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <mitsuba/mitsuba.h>

#include "FLAGS.h"
#include "effmis_structs.h"
#include "effmis_aovs.h"
#include "bidir.h"


MTS_NAMESPACE_BEGIN
class EffMISWorkResult;
class PilotPTWorkResult;

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

struct EffMISConfiguration {
    int maxDepth;
    bool lightImage;
    bool sampleDirect;
    int rrDepth;
    int budget;
    bool perPixelConnections;
    bool usePixelEstimate;
    bool useBidirPilot;
    float outlierFactor; // threshold to clamp based on the pixel estimate

    size_t numConnections;
    size_t pilotIterations;

    inline EffMISConfiguration() {}

    inline EffMISConfiguration(Stream *stream)
    {
        maxDepth = stream->readInt();
        budget = stream->readInt();
        lightImage = stream->readBool();
        sampleDirect = stream->readBool();
        perPixelConnections = stream->readBool();
        usePixelEstimate = stream->readBool();
        useBidirPilot = stream->readBool();
        rrDepth = stream->readInt();
        outlierFactor = stream->readFloat();
        numConnections = stream->readSize();
        pilotIterations = stream->readSize();
    }

    inline void serialize(Stream *stream) const
    {
        stream->writeInt(maxDepth);
        stream->writeInt(budget);
        stream->writeBool(lightImage);
        stream->writeBool(sampleDirect);
        stream->writeBool(perPixelConnections);
        stream->writeBool(usePixelEstimate);
        stream->writeBool(useBidirPilot);
        stream->writeInt(rrDepth);
        stream->writeFloat(outlierFactor);
        stream->writeSize(numConnections);
        stream->writeSize(pilotIterations);
    }
    
    void dump() const
    {
        SLog(EDebug, "Efficiency Aware Bidirectional Path Tracer configuration:");
        SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
        SLog(EDebug, "   Russian roulette depth      : %i", rrDepth);
        SLog(EDebug, "   NEE                         : %s", sampleDirect ? "yes" : "no");
        SLog(EDebug, "   LT                          : %s", lightImage ? "yes" : "no");
        SLog(EDebug, "   Per pixel connections       : %s", perPixelConnections ? "yes" : "no");
        SLog(EDebug, "   Using pixel estimate        : %s", usePixelEstimate ? "yes" : "no");
        SLog(EDebug, "   Outlier factor              : %f", outlierFactor);
        SLog(EDebug, "   * Number of connections     : " SIZE_T_FMT, numConnections);
        SLog(EDebug, "   Pilot iterations            : %i", pilotIterations);
        SLog(EDebug, "   With bidir pilot            : %s", useBidirPilot ? "yes" : "no");
        SLog(EDebug, "   Budget                      : %is", budget);
        SLog(EDebug, "   * -> ignored if pilot iterations > 0");
    }
};

class Path;
/**
 * \brief Stores all configuration parameters of the
 * bidirectional path tracer
 */
struct EffMISContext
{
    EffMISConfiguration config;
    int blockSize;
    Vector2i cropSize;
    size_t sampleCount = 0;

    size_t numConnections = 0; // Current number of connections or maximum number of connections if per-pixel
    size_t internalIterations = 0;

    Float avgCameraPathLength = 0;
    Float avgLightPathLength = 0;

    inline Float getAverageProxyLightPathLength() const {
        if (avgLightPathLength > 0)
            return avgLightPathLength;
        else if (avgCameraPathLength > 0)
            return avgCameraPathLength;
        else
            return 5.0f;
    }

    bool acquireMoments = false;
    bool usePerPixelConnections = false;

    // Assumes these are already sorted
    std::vector<int> connectionCandidates;
    void setupPTCandidates();
    void setupBidirCandidates();

    inline int numberOfPixels() const { return cropSize.x * cropSize.y; }

    void putMomentSample(EffMISWorkResult* wr, const Point2 &samplePos, const Spectrum &weight, const BidirPdfs::ProxyWeights &proxyWeights) const;

    ref<Bitmap> numConnectionsImage;
    inline Float numberOfConnections(const Point2i& pixel) const {
        if (!usePerPixelConnections)
            return numConnections;

        SAssert(numConnectionsImage);
        return numConnectionsImage->getFloatData()[pixel.y * cropSize.x + pixel.x];
    }
    inline Float numberOfConnections(const Point2& pixel) const {
        if (!usePerPixelConnections)
            return numConnections;

        SAssert(numConnectionsImage);
        int x = std::floor(pixel.x);
        int y = std::floor(pixel.y);
        return numConnectionsImage->getFloatData()[y * cropSize.x + x];
    }

    StatsRecursiveImageBlocks *statsImages = nullptr; // AOVs
    ref<Bitmap>* pixelEstimate = nullptr;

    EffMISWorkResult *wr = nullptr;

    EffMISWorkResult **wrs = nullptr; // Pre-allocated EffMISWorkResult memory
    size_t nCores = 0;

    inline EffMISContext() {}

    inline EffMISContext(Stream *stream) : config(stream)
    {
        blockSize = stream->readInt();
        cropSize = Vector2i(stream);
        sampleCount = stream->readSize();
        numConnections = stream->readSize();
    }

    inline void serialize(Stream *stream) const
    {
        config.serialize(stream);
        stream->writeInt(blockSize);
        cropSize.serialize(stream);
        stream->writeSize(sampleCount);
        stream->writeSize(numConnections);
    }

    void dump() const
    {
        config.dump();
    }
};

MTS_NAMESPACE_END
