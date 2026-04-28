#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <unordered_map>

namespace colmap {

// FORK-REMOVAL TODO: entire file is fork-only; vanilla colmap filters
// triangulation angles inside the BA loop. See fork_removal_todo.md.

// Drop track elements whose bearing-vs-3D angle exceeds threshold.
// Uncalibrated cameras use 2x relaxed threshold. Returns shrunk count.
int FilterTracksByAngle(CorrespondenceGraph& view_graph,
                        const Reconstruction& rec,
                        std::unordered_map<point3D_t, Point3D>& tracks,
                        double max_angle_error_deg = 1.);

// Drop tracks with max triangulation angle below threshold. Returns count.
int FilterTrackTriangulationAngle(
    CorrespondenceGraph& view_graph,
    const Reconstruction& rec,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double min_angle_deg = 1.);

}  // namespace colmap
