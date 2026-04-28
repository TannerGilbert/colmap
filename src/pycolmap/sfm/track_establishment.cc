// Binding for EstablishTracksFromCorrGraph, AppendLoopClosureObservations,
// and SubsampleTracks.

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/track.h"
#include "colmap/sfm/track_establishment.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <unordered_map>
#include <unordered_set>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

py::dict RunEstablishFullTracks(CorrespondenceGraph& correspondence_graph,
                                py::dict images_py,
                                const TrackEstablishmentOptions& options,
                                bool lc_second_pass) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  image_id_to_keypoints.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    image_id_to_keypoints.emplace(image_id, image.features);
  }

  std::vector<image_pair_t> valid_pair_ids;
  valid_pair_ids.reserve(correspondence_graph.NumImagePairs());
  for (const auto& [pair_id, image_pair] :
       correspondence_graph.MutableImagePairs()) {
    if (image_pair.is_valid) {
      valid_pair_ids.push_back(pair_id);
    }
  }

  std::unordered_map<point3D_t, Point3D> tracks;
  {
    py::gil_scoped_release release;
    MatchPredicate ignore_match;
    TrackEstablishmentOptions to = options;
    if (lc_second_pass) {
      ignore_match = MakeLoopClosureMatchPredicate(
          valid_pair_ids, correspondence_graph);
      to.required_tracks_per_view = std::numeric_limits<int>::max();
    }
    tracks = EstablishTracksFromCorrGraph(valid_pair_ids,
                                          correspondence_graph,
                                          image_id_to_keypoints,
                                          to,
                                          ignore_match);
    if (lc_second_pass) {
      AppendLoopClosureObservations(
          valid_pair_ids, correspondence_graph, tracks);
    }
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  return tracks_out;
}

py::dict RunFindTracksForProblem(CorrespondenceGraph& /*correspondence_graph*/,
                                 py::dict images_py,
                                 py::dict tracks_full_py,
                                 const TrackSubsampleOptions& options) {
  std::unordered_set<image_t> registered_image_ids;
  std::unordered_map<image_t, std::vector<double>> depth_priors;
  std::unordered_map<image_t, std::vector<bool>> depth_prior_validity;
  for (auto item : images_py) {
    const auto image_id = py::cast<image_t>(item.first);
    const auto image = py::cast<Image>(item.second);
    if (image.is_registered) {
      registered_image_ids.insert(image_id);
    }
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  tracks_full.reserve(tracks_full_py.size());
  for (auto item : tracks_full_py) {
    tracks_full.emplace(py::cast<point3D_t>(item.first),
                        py::cast<Point3D>(item.second));
  }

  std::unordered_map<point3D_t, Point3D> selected;
  {
    py::gil_scoped_release release;
    selected = SubsampleTracks(options,
                               registered_image_ids,
                               depth_priors,
                               depth_prior_validity,
                               tracks_full);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : selected) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  return tracks_out;
}

}  // namespace

void BindTrackEstablishment(py::module& m) {
  auto PyEstOpts =
      py::classh<TrackEstablishmentOptions>(m, "TrackEstablishmentOptions")
          .def(py::init<>())
          .def_readwrite(
              "intra_image_consistency_threshold",
              &TrackEstablishmentOptions::intra_image_consistency_threshold)
          .def_readwrite("min_num_views_per_track",
                         &TrackEstablishmentOptions::min_num_views_per_track)
          .def_readwrite("required_tracks_per_view",
                         &TrackEstablishmentOptions::required_tracks_per_view);
  MakeDataclass(PyEstOpts);

  auto PySubOpts =
      py::classh<TrackSubsampleOptions>(m, "TrackSubsampleOptions")
          .def(py::init<>())
          .def_readwrite("min_num_views_per_track",
                         &TrackSubsampleOptions::min_num_views_per_track)
          .def_readwrite("max_num_views_per_track",
                         &TrackSubsampleOptions::max_num_views_per_track)
          .def_readwrite("required_tracks_per_view",
                         &TrackSubsampleOptions::required_tracks_per_view)
          .def_readwrite("max_num_tracks",
                         &TrackSubsampleOptions::max_num_tracks)
          .def_readwrite("two_view_depth_gate",
                         &TrackSubsampleOptions::two_view_depth_gate);
  MakeDataclass(PySubOpts);

  m.def("establish_full_tracks",
        &RunEstablishFullTracks,
        "correspondence_graph"_a,
        "images"_a,
        "options"_a,
        "lc_second_pass"_a = false,
        "Build tracks from a CorrespondenceGraph + dict-of-images via "
        "the union-find helper. When ``lc_second_pass=True``, "
        "AppendLoopClosureObservations runs after to populate "
        "``Track::lc_elements`` from inliers flagged "
        "``ImagePair::are_lc==true``.");

  m.def("find_tracks_for_problem",
        &RunFindTracksForProblem,
        "correspondence_graph"_a,
        "images"_a,
        "tracks_full"_a,
        "options"_a,
        "Greedy length-sorted subsample of ``tracks_full``. Reads "
        "``Image::is_registered`` from ``images``.");
}
