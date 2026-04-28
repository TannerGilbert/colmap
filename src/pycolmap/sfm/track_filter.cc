// Pycolmap binding for FilterTracksByAngle and FilterTrackTriangulationAngle.
// Bound at top-level ``pycolmap.filter_tracks_by_angle`` /
// ``pycolmap.filter_track_triangulation_angle``.

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/track_filter.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTrackFilter(py::module& m) {
  m.def(
      "filter_tracks_by_angle",
      [](CorrespondenceGraph& correspondence_graph,
         Reconstruction& rec,
         std::unordered_map<point3D_t, Point3D>& tracks,
         double max_angle_error) {
        int counter;
        {
          py::gil_scoped_release release;
          counter = FilterTracksByAngle(
              correspondence_graph, rec, tracks, max_angle_error);
        }
        py::dict output;
        output["tracks"] = tracks;
        output["counter"] = counter;
        return output;
      },
      "correspondence_graph"_a,
      "rec"_a,
      "tracks"_a,
      "max_angle_error"_a = 1.,
      "Drop track elements whose bearing-vs-3D angle exceeds the "
      "threshold (degrees). Calibrated cameras get the supplied "
      "threshold; uncalibrated cameras get a 2x relax. Returns a "
      "dict with keys 'tracks' (filtered subset) and 'counter' "
      "(number of tracks whose element list shrank).");

  m.def(
      "filter_track_triangulation_angle",
      [](CorrespondenceGraph& correspondence_graph,
         Reconstruction& rec,
         std::unordered_map<point3D_t, Point3D>& tracks,
         double min_angle) {
        int counter;
        {
          py::gil_scoped_release release;
          counter = FilterTrackTriangulationAngle(
              correspondence_graph, rec, tracks, min_angle);
        }
        py::dict output;
        output["tracks"] = tracks;
        output["counter"] = counter;
        return output;
      },
      "correspondence_graph"_a,
      "rec"_a,
      "tracks"_a,
      "min_angle"_a = 1.,
      "Drop tracks whose maximum pairwise triangulation angle is "
      "below the threshold (degrees). Mutates the dict in place; "
      "returns 'tracks' (filtered) and 'counter' (number dropped).");
}
