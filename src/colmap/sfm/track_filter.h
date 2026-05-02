#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <unordered_map>

namespace colmap {

// FORK-REMOVAL TODO — this entire file (FilterTracksByAngle +
// FilterTrackTriangulationAngle and the corresponding bodies in
// track_filter.cc) is fork-only. Vanilla colmap does
// triangulation-angle filtering inside the BA / triangulation loop via
// ``ObservationManager::FindPoints3DWithSmallTriangulationAngle``, not
// as a standalone pass. The only consumer is the F-side
// `pycolmap.filter_tracks_by_angle` call from
// `mpsfm/mapper/glomap/phases/global_positioning.py`. Slated for
// removal once the reproducibility window closes; see
// `.claude/notes/glomap_audit/fork_removal_todo.md`.

// Drop ``Track::Elements`` whose bearing-vs-3D-point angle exceeds the
// threshold. Reads ``Image::features_undist`` (precomputed unit ray)
// per element. Calibrated cameras (``Camera::has_prior_focal_length``)
// use ``cos(max_angle_error_deg)``; uncalibrated cameras get a 2x relaxed
// threshold ``cos(2 * max_angle_error_deg)`` since their focal is still
// being optimized. Mutates ``tracks`` in place via ``Track::SetElements``;
// returns the count of tracks whose element list shrank.
int FilterTracksByAngle(CorrespondenceGraph& view_graph,
                        const Reconstruction& rec,
                        std::unordered_map<point3D_t, Point3D>& tracks,
                        double max_angle_error_deg = 1.);

// Drop tracks whose maximum pairwise triangulation angle is below the
// threshold. Mirrors
// ``ObservationManager::FindPoints3DWithSmallTriangulationAngle`` but
// operates on the dict-of-tracks state; shares the angle math via
// ``CalculateTriangulationAngle``. Marks dropped tracks with
// ``Track::SetElements({})``; returns the dropped count.
int FilterTrackTriangulationAngle(
    CorrespondenceGraph& view_graph,
    const Reconstruction& rec,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double min_angle_deg = 1.);

}  // namespace colmap
