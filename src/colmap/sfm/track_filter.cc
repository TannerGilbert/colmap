#include "colmap/sfm/track_filter.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

namespace colmap {
namespace {
constexpr double EPS = 1e-12;
}  // namespace

int FilterTracksByAngle(CorrespondenceGraph& /*view_graph*/,
                        const Reconstruction& rec,
                        std::unordered_map<point3D_t, Point3D>& tracks,
                        double max_angle_error_deg) {
  int counter = 0;
  const double thres = std::cos(DegToRad(max_angle_error_deg));
  const double thres_uncalib = std::cos(DegToRad(max_angle_error_deg * 2));
  for (auto& [track_id, point3D] : tracks) {
    std::vector<TrackElement> elements_new;
    for (const auto& el : point3D.track.Elements()) {
      const Image& image = rec.Image(el.image_id);
      const Eigen::Vector3d& feature_undist =
          image.features_undist.at(el.point2D_idx);
      Eigen::Vector3d pt_calc = image.CamFromWorld() * point3D.xyz;
      if (pt_calc(2) < EPS) continue;

      pt_calc = pt_calc.normalized();
      const double thres_cam =
          rec.Camera(image.CameraId()).has_prior_focal_length ? thres
                                                              : thres_uncalib;

      if (pt_calc.dot(feature_undist) > thres_cam) {
        elements_new.emplace_back(el);
      }
    }
    if (elements_new.size() != point3D.track.Length()) {
      counter++;
      point3D.track.SetElements(std::move(elements_new));
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by angle error";
  return counter;
}

int FilterTrackTriangulationAngle(CorrespondenceGraph& /*view_graph*/,
                                  const Reconstruction& rec,
                                  std::unordered_map<point3D_t, Point3D>& tracks,
                                  double min_angle_deg) {
  int counter = 0;
  const double min_angle_rad = DegToRad(min_angle_deg);
  std::unordered_map<image_t, Eigen::Vector3d> proj_centers;
  for (auto& [track_id, point3D] : tracks) {
    bool keep_point = false;
    const auto& elements = point3D.track.Elements();
    for (size_t i1 = 0; i1 < elements.size() && !keep_point; ++i1) {
      const image_t image_id1 = elements[i1].image_id;
      auto it1 = proj_centers.find(image_id1);
      if (it1 == proj_centers.end()) {
        const Rigid3d cfw = rec.Image(image_id1).CamFromWorld();
        it1 = proj_centers
                  .emplace(image_id1,
                           cfw.rotation().inverse() * -cfw.translation())
                  .first;
      }
      const Eigen::Vector3d& proj_center1 = it1->second;
      for (size_t i2 = 0; i2 < i1; ++i2) {
        const Eigen::Vector3d& proj_center2 =
            proj_centers.at(elements[i2].image_id);
        const double tri_angle = CalculateTriangulationAngle(
            proj_center1, proj_center2, point3D.xyz);
        if (tri_angle >= min_angle_rad) {
          keep_point = true;
          break;
        }
      }
    }

    if (!keep_point) {
      counter++;
      point3D.track.SetElements({});
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace colmap
