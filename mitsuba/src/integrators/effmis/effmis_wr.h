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

#include <mitsuba/render/imageblock.h>
#include <mitsuba/core/fresolver.h>

#include "effmis.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

/**
   Bidirectional path tracing needs its own WorkResult implementation,
   since each rendering thread simultaneously renders to a small 'camera
   image' block and potentially a full-resolution 'light image'.
   This is once per thread
*/
class EffMISWorkResult : public WorkResult
{
public:
    EffMISWorkResult(int workerIndex, const EffMISContext &conf, const ReconstructionFilter *filter, Vector2i blockSize = Vector2i(-1, -1));

    // Clear the contents of the work result
    void clearBlock();

    // Clear the contents of the work result
    void clearMoments();

    /// Fill the work result with content acquired from a binary data stream
    void load(Stream *stream) override;

    /// Serialize a work result to a binary data stream
    void save(Stream *stream) const override;

    /// Aaccumulate another work result into this one
    void putBlock(const EffMISWorkResult *workResult);
    void putMomentBlocks(const EffMISWorkResult *workResult);
    void scaleMomentBlocks(Float scale);

#if EFFMIS_BDPT_DEBUG == 1
    void putDebugBlock(const EffMISWorkResult *workResult);
    void clearDebugBlocks();
    void checkMIS(const EffMISContext& conf, Float scale);

    /* In debug mode, this function allows to dump the contributions of
       the individual sampling strategies to a series of images */
    void dumpDebug(Scene* scene, const EffMISContext& conf, const fs::path &prefix, const fs::path &stem, float weight) const;

    inline void putDebugSample(int s, int t, const Point2 &sample, const Spectrum &value, Float misWeight)
    {
        Assert(value.isValid());
        const int index = strategyIndex(s, t);
        m_rawBlocks[index]->put(sample, value, 1.0f);
        m_misBlocks[index]->put(sample, Spectrum(misWeight), 1.0f);
    }
#endif

    inline void putSample(const Point2 &sample, const Spectrum &spec)
    {
        m_block->put(sample, spec, 1.0f);
    }

    inline const ImageBlock *getImageBlock() const
    {
        return m_block.get();
    }

    inline void setSize(const Vector2i &size)
    {
        m_block->setSize(size);
    }

    inline void setOffset(const Point2i &offset)
    {
        m_block->setOffset(offset);
    }

    void clearLightImage();
    void putLightImage(const EffMISWorkResult *workResult);

    inline void putLightSample(const Point2 &sample, const Spectrum &spec)
    {
        m_lightImage->put(sample, spec, 1.0f);
    }

    inline const ImageBlock *getLightImage() const
    {
        return m_lightImage.get();
    }

    inline void putMomentSample(int index, const Point2 &sample, Float value)
    {
        m_candidateMoments[index]->put(sample, Spectrum(value), 1.0f);
    }

    inline const ImageBlock* getMoments(size_t index) const
    {
        return m_candidateMoments[index].get();
    }

    inline ImageBlock* getMoments(size_t index)
    {
        return m_candidateMoments[index].get();
    }

    void dumpMoments(Scene* scene, const EffMISContext &conf, const fs::path &prefix, const fs::path &stem, float weight) const;

    inline int workerIndex() const { return m_workerIndex; }

    /// Return a string representation
    std::string toString() const;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~EffMISWorkResult();

    inline int strategyIndex(int s, int t) const
    {
        int above = s + t - 2;
        return s + above * (5 + above) / 2;
    }

protected:
    const int m_workerIndex;

#if EFFMIS_BDPT_DEBUG == 1
    ref_vector<ImageBlock> m_rawBlocks;
    ref_vector<ImageBlock> m_misBlocks;
#endif
    ref<ImageBlock> m_block;
    ref<ImageBlock> m_lightImage;
    ref_vector<ImageBlock> m_candidateMoments;
};
MTS_NAMESPACE_END
