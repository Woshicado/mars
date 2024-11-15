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

#if !defined(__BMARS_H)
#define __BMARS_H

#include <mitsuba/mitsuba.h>

#include "FLAGS.h"
#include "ears_structs.h"
#include "octtree.h"
#include "bmars_aovs.h"

/**
 * When the following is set to "1", the Bidirectional Path Tracer
 * will generate a series of debugging images that split up the final
 * rendering into the weighted contributions of the individual sampling
 * strategies.
 */
// #define BDPT_DEBUG 1

MTS_NAMESPACE_BEGIN
class BMARSWorkResult;
struct ImageStatistics;

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/**
 * \brief Stores all configuration parameters of the
 * bidirectional path tracer
 */
struct BMARSConfiguration {
    /// Standard information
    int blockSize, borderSize;
    Vector2i cropSize;
    bool lightImage;
    bool sampleDirect;
    bool showWeighted;
    bool strictNormals;
    bool hideEmitters;
    bool splitBSDF;
    bool splitNEE;
    bool splitLP;
    bool disableLP;
    bool shareSF;
    bool shareLP;

    /// Necessary to parse RRSMethod
    std::string rrsStr;
    Float splittingMin;
    Float splittingMax;
    Float sfExp;
    int maxDepth;

    /// Information from the outside for the workers
    StatsRecursiveImageBlocks* statsImages; // AOVs
    ImageStatistics *imageStatistics;       // Estimates
    ref<Bitmap>* pixelEstimate;
    BMARSWorkResult** wrs;  // Pre-allocated BMARSWorkResult memory
    Float imageEarsFactor;
    Octtree *cache;
    bool budgetAware;
    int outlierRejection;

    /// RRSMethod to query and use
    RRSMethod* renderRRSMethod;
    RRSMethod* currentRRSMethod;


    inline BMARSConfiguration() { }

    inline BMARSConfiguration(Stream *stream) {
        maxDepth = stream->readInt();
        blockSize = stream->readInt();
        lightImage = stream->readBool();
        sampleDirect = stream->readBool();
        showWeighted = stream->readBool();
        cropSize = Vector2i(stream);
    }

    inline void serialize(Stream *stream) const {
        stream->writeInt(maxDepth);
        stream->writeInt(blockSize);
        stream->writeBool(lightImage);
        stream->writeBool(sampleDirect);
        stream->writeBool(showWeighted);
        cropSize.serialize(stream);
    }

    void dump() const {
        SLog(EDebug, "Bidirectional path tracer configuration:");
        SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
        SLog(EDebug, "   Image size                  : %ix%i",
            cropSize.x, cropSize.y);
        SLog(EDebug, "   Direct sampling strategies  : %s",
            sampleDirect ? "yes" : "no");
        SLog(EDebug, "   Generate light image        : %s",
            lightImage ? "yes" : "no");
        SLog(EDebug, "   Block size                  : %i", blockSize);
        #if BDPT_DEBUG == 1
            SLog(EDebug, "   Show weighted contributions : %s", showWeighted ? "yes" : "no");
        #endif
    }
};

MTS_NAMESPACE_END

#endif /* __BMARS_H */
