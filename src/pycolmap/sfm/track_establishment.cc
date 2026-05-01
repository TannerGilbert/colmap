// Binding for EstablishTracksFromCorrGraph, AppendLoopClosureObservations,
// and FilterTracksForProblem.

#include "colmap/sfm/track_establishment.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/track.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <limits>
#include <unordered_map>
#include <unordered_set>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

std::vector<image_pair_t> CollectValidPairIds(
    CorrespondenceGraph& correspondence_graph) {
  std::vector<image_pair_t> pair_ids;
  pair_ids.reserve(correspondence_graph.NumImagePairs());
  for (const auto& [pair_id, image_pair] :
       correspondence_graph.MutableImagePairs()) {
    if (image_pair.is_valid) {
      pair_ids.push_back(pair_id);
    }
  }
  return pair_ids;
}

py::dict RunEstablishFullTracks(CorrespondenceGraph& correspondence_graph,
                                py::dict images_py,
                                const TrackEstablishmentOptions& options,
                                bool lc_second_pass,
                                CorrespondenceGraph* lc_correspondence_graph) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first), py::cast<Image>(item.second));
  }

  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  image_id_to_keypoints.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    image_id_to_keypoints.emplace(image_id, image.features);
  }

  const std::vector<image_pair_t> valid_pair_ids =
      CollectValidPairIds(correspondence_graph);

  std::unordered_map<point3D_t, Point3D> tracks;
  {
    py::gil_scoped_release release;
    TrackEstablishmentOptions to = options;
    MatchPredicate ignore_match;
    if (lc_second_pass) {
      to.required_tracks_per_view = std::numeric_limits<int>::max();
      CorrespondenceGraph& lc_cg = lc_correspondence_graph
                                       ? *lc_correspondence_graph
                                       : correspondence_graph;
      const std::vector<image_pair_t> lc_pair_ids = CollectValidPairIds(lc_cg);
      ignore_match = MakeLoopClosureMatchPredicate(lc_pair_ids, lc_cg);
    }
    tracks = EstablishTracksFromCorrGraph(valid_pair_ids,
                                          correspondence_graph,
                                          image_id_to_keypoints,
                                          to,
                                          ignore_match);
    if (lc_second_pass) {
      CorrespondenceGraph& lc_cg = lc_correspondence_graph
                                       ? *lc_correspondence_graph
                                       : correspondence_graph;
      const std::vector<image_pair_t> lc_pair_ids = CollectValidPairIds(lc_cg);
      AppendLoopClosureObservations(lc_pair_ids, lc_cg, tracks);
    }
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  return tracks_out;
}

py::dict RunFindTracksForProblem(py::dict images_py,
                                 py::dict tracks_full_py,
                                 const TrackProblemFilterOptions& options) {
  std::unordered_set<image_t> registered_image_ids;
  for (auto item : images_py) {
    registered_image_ids.insert(py::cast<image_t>(item.first));
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
    selected =
        FilterTracksForProblem(options, registered_image_ids, tracks_full);
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
      py::classh<TrackProblemFilterOptions>(m, "TrackProblemFilterOptions")
          .def(py::init<>())
          .def_readwrite("min_num_views_per_track",
                         &TrackProblemFilterOptions::min_num_views_per_track)
          .def_readwrite("max_num_views_per_track",
                         &TrackProblemFilterOptions::max_num_views_per_track);
  MakeDataclass(PySubOpts);

  m.def("establish_full_tracks",
        &RunEstablishFullTracks,
        "correspondence_graph"_a,
        "images"_a,
        "options"_a,
        "lc_second_pass"_a = false,
        "lc_correspondence_graph"_a = nullptr,
        "Build tracks from a CorrespondenceGraph + dict-of-images via "
        "the union-find helper. When ``lc_second_pass=True``, "
        "AppendLoopClosureObservations runs after to populate "
        "``Track::lc_elements`` from inliers flagged "
        "``ImagePair::are_lc==true`` in ``lc_correspondence_graph`` "
        "(falls back to ``correspondence_graph`` if not provided).");

  m.def("find_tracks_for_problem",
        &RunFindTracksForProblem,
        "images"_a,
        "tracks_full"_a,
        "options"_a,
        "Filter ``tracks_full`` for the optimization problem. Reads "
        "registered image ids from the keys of the filtered ``images`` dict.");
}
