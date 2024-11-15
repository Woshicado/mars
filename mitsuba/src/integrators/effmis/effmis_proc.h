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

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>

#include "effmis_wr.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

/**
 * \brief Renders work units (rectangular image regions) using
 * bidirectional path tracing
 */
class EffMISProcess : public BlockedRenderProcess {
public:
    EffMISProcess(const RenderJob *parent, RenderQueue *queue,
        const EffMISContext &config, EffMISWorkResult* wr);

    /// Develop the image
    void developLightImage(int spp);

    Float getAverageCameraPathLength() const;
    Float getAverageLightPathLength() const;

    /* ParallelProcess impl. */
    void processResult(const WorkResult *wr, bool cancelled) override;
    ref<WorkProcessor> createWorkProcessor() const override;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~EffMISProcess() { }
private:
    ref<EffMISWorkResult> m_result;
    const EffMISContext& m_context;
};

MTS_NAMESPACE_END
