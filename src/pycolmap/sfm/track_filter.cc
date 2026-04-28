// Pycolmap binding for the dict-of-Point3D track-filter free
// functions in ``colmap/sfm/track_filter.{h,cc}``. Bound at top-level
// ``pycolmap.filter_tracks_by_angle`` / ``pycolmap.filter_track_triangulation_angle``.

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/sfm/track_filter.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

py::dict RunFilterTracksByAngle(CorrespondenceGraph& correspondence_graph,
                                py::dict cameras_py,
                                py::dict images_py,
                                py::dict tracks_py,
                                double max_angle_error_deg) {
  std::unordered_map<camera_t, Camera> cameras;
  cameras.reserve(cameras_py.size());
  for (auto item : cameras_py) {
    cameras.emplace(py::cast<camera_t>(item.first),
                    py::cast<Camera>(item.second));
  }
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }
  std::unordered_map<point3D_t, Point3D> tracks;
  tracks.reserve(tracks_py.size());
  for (auto item : tracks_py) {
    tracks.emplace(py::cast<point3D_t>(item.first),
                   py::cast<Point3D>(item.second));
  }

  int counter;
  {
    py::gil_scoped_release release;
    counter = FilterTracksByAngle(
        correspondence_graph, cameras, images, tracks, max_angle_error_deg);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  py::dict output;
  output["tracks"] = tracks_out;
  output["counter"] = counter;
  return output;
}

py::dict RunFilterTrackTriangulationAngle(CorrespondenceGraph& correspondence_graph,
                                          py::dict images_py,
                                          py::dict tracks_py,
                                          double min_angle_deg) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }
  std::unordered_map<point3D_t, Point3D> tracks;
  tracks.reserve(tracks_py.size());
  for (auto item : tracks_py) {
    tracks.emplace(py::cast<point3D_t>(item.first),
                   py::cast<Point3D>(item.second));
  }

  int counter;
  {
    py::gil_scoped_release release;
    counter = FilterTrackTriangulationAngle(
        correspondence_graph, images, tracks, min_angle_deg);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  py::dict output;
  output["tracks"] = tracks_out;
  output["counter"] = counter;
  return output;
}

}  // namespace

void BindTrackFilter(py::module& m) {
  m.def("filter_tracks_by_angle",
        &RunFilterTracksByAngle,
        "correspondence_graph"_a,
        "cameras"_a,
        "images"_a,
        "tracks"_a,
        "max_angle_error"_a = 1.,
        "Drop track elements whose bearing-vs-3D angle exceeds the "
        "threshold (degrees). Calibrated cameras get the supplied "
        "threshold; uncalibrated cameras get a 2x relax. Returns a "
        "dict with keys 'tracks' (filtered subset) and 'counter' "
        "(number of tracks whose element list shrank).");
  m.def("filter_track_triangulation_angle",
        &RunFilterTrackTriangulationAngle,
        "correspondence_graph"_a,
        "images"_a,
        "tracks"_a,
        "min_angle"_a = 1.,
        "Drop tracks whose maximum pairwise triangulation angle is "
        "below the threshold (degrees). Mutates the dict in place; "
        "returns 'tracks' (filtered) and 'counter' (number dropped).");
}
