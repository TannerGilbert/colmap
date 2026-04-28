// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include <unordered_map>

namespace colmap {

// Drop ``Track::Elements`` whose bearing-vs-3D-point angle exceeds the
// threshold. Reads ``Image::features_undist`` (precomputed unit ray)
// per element. Calibrated cameras (``Camera::has_prior_focal_length``)
// use ``cos(max_angle_error_deg)``; uncalibrated cameras get a 2x relaxed
// threshold ``cos(2 * max_angle_error_deg)`` since their focal is still
// being optimized. Mutates ``tracks`` in place via ``Track::SetElements``;
// returns the count of tracks whose element list shrank.
int FilterTracksByAngle(
    CorrespondenceGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double max_angle_error_deg = 1.);

// Drop tracks whose maximum pairwise triangulation angle is below the
// threshold. Mirrors
// ``ObservationManager::FindPoints3DWithSmallTriangulationAngle`` but
// operates on the dict-of-tracks state used here instead of a
// ``Reconstruction``; shares the angle math via
// ``CalculateTriangulationAngle``. Marks dropped tracks with
// ``Track::SetElements({})``; returns the dropped count.
int FilterTrackTriangulationAngle(
    CorrespondenceGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double min_angle_deg = 1.);

}  // namespace colmap
