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

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/plugin.h>
#include "effmis_wr.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

EffMISWorkResult::EffMISWorkResult(int workerIndex, const EffMISContext &ctx, const ReconstructionFilter *rfilter, Vector2i blockSize)
    : WorkResult()
    , m_workerIndex(workerIndex)
{
    /* Stores the 'camera image' -- this can be blocked when
       spreading out work to multiple workers */
    if (blockSize == Vector2i(-1, -1))
        blockSize = Vector2i(ctx.blockSize, ctx.blockSize);

    m_block = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter);
    m_block->setOffset(Point2i(0, 0));
    m_block->setSize(blockSize);

    /* When debug mode is active, we additionally create
       full-resolution bitmaps storing the contributions of
       each individual sampling strategy */
#if EFFMIS_BDPT_DEBUG == 1
    m_rawBlocks.resize(
        ctx.config.maxDepth * (5 + ctx.config.maxDepth) / 2);

    for (size_t i = 0; i < m_rawBlocks.size(); ++i)
    {
        m_rawBlocks[i] = new ImageBlock(
            Bitmap::ESpectrum, ctx.cropSize, rfilter);
        m_rawBlocks[i]->setOffset(Point2i(0, 0));
        m_rawBlocks[i]->setSize(ctx.cropSize);
    }

    m_misBlocks.resize(
        ctx.config.maxDepth * (5 + ctx.config.maxDepth) / 2);

    for (size_t i = 0; i < m_misBlocks.size(); ++i)
    {
        m_misBlocks[i] = new ImageBlock(
            Bitmap::ELuminance, ctx.cropSize, rfilter);
        m_misBlocks[i]->setOffset(Point2i(0, 0));
        m_misBlocks[i]->setSize(ctx.cropSize);
    }
#endif

    if (ctx.config.lightImage)
    {
        /* Stores the 'light image' -- every worker requires a
           full-resolution version, since contributions of s==0
           and s==1 paths can affect any pixel of this bitmap */
        m_lightImage = new ImageBlock(Bitmap::ESpectrum, ctx.cropSize, rfilter);
        m_lightImage->setSize(ctx.cropSize);
        m_lightImage->setOffset(Point2i(0, 0));
    }

    m_candidateMoments.resize(ctx.connectionCandidates.size());
    for (size_t i = 0; i < m_candidateMoments.size(); ++i)
    {
        m_candidateMoments[i] = new ImageBlock(Bitmap::ELuminance, ctx.cropSize, rfilter);
        m_candidateMoments[i]->setOffset(Point2i(0, 0));
        m_candidateMoments[i]->setSize(ctx.cropSize);
    }
}

EffMISWorkResult::~EffMISWorkResult() {}

void EffMISWorkResult::putBlock(const EffMISWorkResult *workResult)
{
    m_block->put(workResult->m_block.get());
}

void EffMISWorkResult::clearBlock()
{
    m_block->clear();
}

void EffMISWorkResult::clearMoments()
{
    for (size_t i = 0; i < m_candidateMoments.size(); ++i)
        m_candidateMoments[i]->clear();
}

void EffMISWorkResult::putMomentBlocks(const EffMISWorkResult *workResult)
{
    for (size_t i = 0; i < m_candidateMoments.size(); ++i)
        m_candidateMoments[i]->put(workResult->getMoments(i));
}

void EffMISWorkResult::scaleMomentBlocks(Float scale)
{
    for (size_t i = 0; i < m_candidateMoments.size(); ++i) // Does not care about regions/blocks
        m_candidateMoments[i]->getBitmap()->scale(scale);
}

void EffMISWorkResult::putLightImage(const EffMISWorkResult *workResult)
{
    Assert(m_lightImage);
    m_lightImage->put(workResult->m_lightImage.get());
}

void EffMISWorkResult::clearLightImage()
{
    if (!m_lightImage)
        return;

    m_lightImage->clear();
}

#if EFFMIS_BDPT_DEBUG == 1
void EffMISWorkResult::putDebugBlock(const EffMISWorkResult *workResult) {
    for (size_t i = 0; i < m_rawBlocks.size(); ++i) {
        m_rawBlocks[i]->put(workResult->m_rawBlocks[i].get());
        m_misBlocks[i]->put(workResult->m_misBlocks[i].get());
    }
}

void EffMISWorkResult::clearDebugBlocks() {
    for (size_t i = 0; i < m_rawBlocks.size(); ++i) {
        m_rawBlocks[i]->clear();
        m_misBlocks[i]->clear();
    }
}

/* In debug mode, this function allows to dump the contributions of
   the individual sampling strategies to a series of images */
void EffMISWorkResult::dumpDebug(Scene* scene, const EffMISContext& conf, const fs::path &prefix, const fs::path &stem, float weight) const
{
    std::stringstream types;
    std::stringstream names;

    for (int k = 1; k <= conf.config.maxDepth; ++k)
    {
        for (int t = 0; t <= k + 1; ++t)
        {
            const size_t s = k + 1 - t;
            if(k > 0) {
                types << ", ";
                names << ", ";
            }

            types << "luminance, rgb";
            names << "s" << s << "_t" << t << ".mis, s" << s << "_t" << t << ".raw";
        }
    }

    ref<Film> film = scene->getFilm();
    auto properties = Properties(film->getProperties());
    properties.setString("pixelFormat", types.str());
    properties.setString("channelNames", names.str());
    auto rfilter = film->getReconstructionFilter();

    const auto outputFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
    outputFilm->addChild(rfilter);
    outputFilm->configure();

    const auto outputImage = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, film->getCropSize(), nullptr, m_rawBlocks.size() * 2 * SPECTRUM_SAMPLES + 2);

    Float *outputData = outputImage->getBitmap()->getFloatData();
    for (int y = 0; y < conf.cropSize.y; ++y) 
    {
        for (int x = 0; x < conf.cropSize.x; ++x)
        {
            Point2i pos = Point2i(x, y);

            for (int k = 1; k <= conf.config.maxDepth; ++k)
            {
                for (int t = 0; t <= k + 1; ++t)
                {
                    const size_t s = k + 1 - t;
                    const size_t index = strategyIndex(s, t);

                    Spectrum v = m_misBlocks[index]->getBitmap()->getPixel(pos);
                    for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                        *(outputData++) = v[i] * weight;
                    
                    v = m_rawBlocks[index]->getBitmap()->getPixel(pos);
                    for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                        *(outputData++) = v[i] * weight;
                }
            }

            *(outputData++) = 1.0f;
            *(outputData++) = 1.0f;
        }
    }
    outputFilm->setBitmap(outputImage->getBitmap());

    std::string suffix = "-bdpt-EffMIS";
    fs::path destPath = scene->getDestinationFile();
    fs::path outPath = destPath.parent_path() / (destPath.leaf().string() + suffix + ".exr");

    outputFilm->setDestinationFile(outPath, 0);
    outputFilm->develop(scene, 0.0f);
}

void EffMISWorkResult::checkMIS(const EffMISContext& conf, Float scale) 
{
    for (int k = 1; k <= conf.config.maxDepth; ++k)
    {
        size_t samples = 0;
        Float totalImageMIS = 0;
        Float minImageMIS = std::numeric_limits<Float>::max();
        Float maxImageMIS = -std::numeric_limits<Float>::max();

        for (int y = 0; y < conf.cropSize.y; ++y) 
        {
            for (int x = 0; x < conf.cropSize.x; ++x)
            {
                const Point2i pos = Point2i(x, y);

                Float totalMIS = 0;
                for (int t = 0; t <= k + 1; ++t)
                {
                    const size_t s = k + 1 - t;
                    const size_t index = strategyIndex(s, t);

                    totalMIS += m_misBlocks[index]->getBitmap()->getPixel(pos).average() * scale;
                }

                if (totalMIS > 0) {
                    ++samples;
                    totalImageMIS += totalMIS;
                    minImageMIS = std::min(minImageMIS, totalMIS);
                    maxImageMIS = std::max(maxImageMIS, totalMIS);
                }
            }
        }

        if (samples > 0)
            Log(EInfo, "[%i] MIS Sum over image [Avg=%f, Min=%f, Max=%f]", k, totalImageMIS / samples, minImageMIS, maxImageMIS);
        else
            Log(EInfo, "[%i] Path length has no MIS", k);
    }
}
#endif

void EffMISWorkResult::load(Stream *stream)
{
#if EFFMIS_BDPT_DEBUG == 1
    for (size_t i = 0; i < m_rawBlocks.size(); ++i) {
        m_rawBlocks[i]->load(stream);
        m_misBlocks[i]->load(stream);
    }
#endif
    for (size_t i = 0; i < m_candidateMoments.size(); ++i)
        m_candidateMoments[i]->load(stream);
    m_block->load(stream);
    if (m_lightImage.get())
        m_lightImage->load(stream);
}

void EffMISWorkResult::save(Stream *stream) const
{
#if EFFMIS_BDPT_DEBUG == 1
    for (size_t i = 0; i < m_rawBlocks.size(); ++i) {
        m_rawBlocks[i]->save(stream);
        m_misBlocks[i]->save(stream);
    }
#endif
    for (size_t i = 0; i < m_candidateMoments.size(); ++i)
        m_candidateMoments[i]->save(stream);
    m_block->save(stream);
    if (m_lightImage.get())
        m_lightImage->save(stream);
}

std::string EffMISWorkResult::toString() const
{
    return m_block->toString();
}

void EffMISWorkResult::dumpMoments(Scene* scene, const EffMISContext &conf, const fs::path &prefix, const fs::path &stem, float weight) const
{
    std::stringstream types;
    std::stringstream names;

    for (size_t index = 0; index  < conf.connectionCandidates.size(); ++index) {
        const int c = conf.connectionCandidates[index];

        if (index > 0) {
            types << ", ";
            names << ", ";
        }

        types << "luminance";
        names << "c" << c;
    }

    ref<Film> film = scene->getFilm();
    auto properties = Properties(film->getProperties());
    properties.setString("pixelFormat", types.str());
    properties.setString("channelNames", names.str());
    auto rfilter = film->getReconstructionFilter();

    const auto outputFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
    outputFilm->addChild(rfilter);
    outputFilm->configure();

    const auto outputImage = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, film->getCropSize(), nullptr, conf.connectionCandidates.size() * SPECTRUM_SAMPLES + 2);

    Float *outputData = outputImage->getBitmap()->getFloatData();
    for (int y = 0; y < conf.cropSize.y; ++y) 
    {
        for (int x = 0; x < conf.cropSize.x; ++x)
        {
            Point2i pos = Point2i(x, y);

            for (size_t index = 0; index < conf.connectionCandidates.size(); ++index)
            {
                const Spectrum v = m_candidateMoments[index]->getBitmap()->getPixel(pos);
                for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                    *(outputData++) = v[i] * weight;
            }

            *(outputData++) = 1.0f;
            *(outputData++) = 1.0f;
        }
    }
    outputFilm->setBitmap(outputImage->getBitmap());

    std::string suffix = "-candidates-EffMIS";
    fs::path destPath = scene->getDestinationFile();
    fs::path outPath = destPath.parent_path() / (destPath.leaf().string() + suffix + ".exr");

    outputFilm->setDestinationFile(outPath, 0);
    outputFilm->develop(scene, 0.0f);
}

MTS_IMPLEMENT_CLASS(EffMISWorkResult, false, WorkResult)
MTS_NAMESPACE_END
