#include "colmap/sfm/track_filter.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/track.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Build a simple PINHOLE camera. ``has_prior`` controls the calibrated/uncalib
// branch in FilterTracksByAngle (calibrated => threshold; uncalibrated => 2x).
Camera MakePinholeCamera(camera_t camera_id, bool has_prior) {
  Camera camera = Camera::CreateFromModelName(camera_id,
                                              "PINHOLE",
                                              /*focal_length=*/100.0,
                                              /*width=*/200,
                                              /*height=*/200);
  camera.has_prior_focal_length = has_prior;
  return camera;
}

// Build a Reconstruction with one camera and a list of (image_id,
// cam_from_world, num_features) images.
Reconstruction MakeRec(
    const Camera& camera,
    const std::vector<std::tuple<image_t, Rigid3d, size_t>>& img_specs) {
  Reconstruction rec;
  rec.AddCameraWithTrivialRig(camera);
  for (const auto& [image_id, pose, num_features] : img_specs) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    image.features_undist.assign(num_features, Eigen::Vector3d::UnitZ());
    rec.AddImageWithTrivialFrame(std::move(image), pose);
  }
  return rec;
}

// Build a Point3D with track over (image_id, point2D_idx) entries.
Point3D MakePoint3D(const Eigen::Vector3d& xyz,
                    const std::vector<std::pair<image_t, point2D_t>>& obs) {
  Point3D point3D;
  point3D.xyz = xyz;
  for (const auto& [image_id, p2d_idx] : obs) {
    point3D.track.AddElement(image_id, p2d_idx);
  }
  return point3D;
}

// === FilterTracksByAngle tests ===

// 2 calibrated cameras both at origin looking +Z. Feature_undist points are
// the back-projected rays of the 3D point => angle error = 0.
// max_angle_error = 1deg should keep both observations.
TEST(TrackFilter, FilterByAngle_AlignedRaysKept) {
  CorrespondenceGraph view_graph;
  std::unordered_map<point3D_t, Point3D> tracks;

  Camera cam = MakePinholeCamera(1, /*has_prior=*/true);

  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-1.0, 0.0, 0.0));
  Reconstruction rec =
      MakeRec(cam, {{1, Rigid3d(), 1}, {2, cam2_from_world, 1}});

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  // Set features_undist to the normalized rays of the projection in each cam.
  rec.Image(1).features_undist[0] =
      (rec.Image(1).CamFromWorld() * xyz).normalized();
  rec.Image(2).features_undist[0] =
      (rec.Image(2).CamFromWorld() * xyz).normalized();

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched =
      FilterTracksByAngle(view_graph, rec, tracks, /*max_angle_error=*/1.0);
  EXPECT_EQ(touched, 0);
  EXPECT_EQ(tracks.at(1).track.Length(), 2u);
}

// Rays misaligned by 5deg from the back-projection => above 1deg threshold
// for calibrated cameras => both observations dropped.
TEST(TrackFilter, FilterByAngle_MisalignedRaysDropped) {
  CorrespondenceGraph view_graph;
  std::unordered_map<point3D_t, Point3D> tracks;

  Camera cam = MakePinholeCamera(1, /*has_prior=*/true);
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-1.0, 0.0, 0.0));
  Reconstruction rec =
      MakeRec(cam, {{1, Rigid3d(), 1}, {2, cam2_from_world, 1}});

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  // Apply a 5deg rotation (around Y) to the back-projected ray to emulate
  // observed feature_undist that is 5deg off the ground-truth ray.
  const double angle_rad = DegToRad(5.0);
  Eigen::AngleAxisd tilt(angle_rad, Eigen::Vector3d::UnitY());
  rec.Image(1).features_undist[0] =
      (tilt * (rec.Image(1).CamFromWorld() * xyz)).normalized();
  rec.Image(2).features_undist[0] =
      (tilt * (rec.Image(2).CamFromWorld() * xyz)).normalized();

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched =
      FilterTracksByAngle(view_graph, rec, tracks, /*max_angle_error=*/1.0);
  EXPECT_EQ(touched, 1);
  EXPECT_EQ(tracks.at(1).track.Length(), 0u);
}

// Same 5deg misalignment, but camera is uncalibrated => threshold doubles to
// 10deg => both observations kept.
TEST(TrackFilter, FilterByAngle_UncalibratedCameraDoubleThreshold) {
  CorrespondenceGraph view_graph;
  std::unordered_map<point3D_t, Point3D> tracks;

  // Uncalibrated camera => threshold becomes 2 * max_angle_error.
  Camera cam_uncalib = MakePinholeCamera(1, /*has_prior=*/false);
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-1.0, 0.0, 0.0));
  Reconstruction rec =
      MakeRec(cam_uncalib, {{1, Rigid3d(), 1}, {2, cam2_from_world, 1}});

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  // 5deg misalignment is above 3deg (calibrated) but below 6deg (2x uncalib).
  const double angle_rad = DegToRad(5.0);
  Eigen::AngleAxisd tilt(angle_rad, Eigen::Vector3d::UnitY());
  rec.Image(1).features_undist[0] =
      (tilt * (rec.Image(1).CamFromWorld() * xyz)).normalized();
  rec.Image(2).features_undist[0] =
      (tilt * (rec.Image(2).CamFromWorld() * xyz)).normalized();

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched =
      FilterTracksByAngle(view_graph, rec, tracks, /*max_angle_error=*/3.0);
  EXPECT_EQ(touched, 0);
  EXPECT_EQ(tracks.at(1).track.Length(), 2u);

  // Sanity: same setup, but now flip the camera to calibrated => 5deg drops.
  Camera cam_calib = MakePinholeCamera(1, /*has_prior=*/true);
  // Replace camera in rec.
  rec.Camera(1) = cam_calib;
  std::unordered_map<point3D_t, Point3D> tracks_calib;
  tracks_calib.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));
  const int touched_calib = FilterTracksByAngle(
      view_graph, rec, tracks_calib, /*max_angle_error=*/3.0);
  EXPECT_EQ(touched_calib, 1);
  EXPECT_EQ(tracks_calib.at(1).track.Length(), 0u);
}

// Point behind camera (cam-frame z < EPS) => observation is skipped (the
// implementation does ``continue`` so it is dropped from the new track).
// Verify only the behind-camera obs is dropped; valid one stays.
TEST(TrackFilter, FilterByAngle_PointBehindCameraSkipped) {
  CorrespondenceGraph view_graph;
  std::unordered_map<point3D_t, Point3D> tracks;

  Camera cam = MakePinholeCamera(1, /*has_prior=*/true);

  // Image 2 looks -Z by rotating 180deg around Y; same world point lands at
  // cam-frame z = -5 (behind camera).
  Eigen::Quaterniond q_flip(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
  Rigid3d cam2_from_world(q_flip, Eigen::Vector3d::Zero());
  Reconstruction rec =
      MakeRec(cam, {{1, Rigid3d(), 1}, {2, cam2_from_world, 1}});

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  rec.Image(1).features_undist[0] =
      (rec.Image(1).CamFromWorld() * xyz).normalized();
  // For img2, give a perfectly aligned feature_undist regardless: the
  // ``z < EPS`` early continue kicks in before the angle check.
  rec.Image(2).features_undist[0] = Eigen::Vector3d::UnitZ();

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched =
      FilterTracksByAngle(view_graph, rec, tracks, /*max_angle_error=*/1.0);
  EXPECT_EQ(touched, 1);
  // Only image 1's observation survived.
  ASSERT_EQ(tracks.at(1).track.Length(), 1u);
  EXPECT_EQ(tracks.at(1).track.Element(0).image_id, 1u);
}

// === FilterTrackTriangulationAngle tests ===

// Build a rec with no camera (FilterTrackTriangulationAngle doesn't use
// cameras) just images.
Reconstruction MakeRecImagesOnly(
    const std::vector<std::pair<image_t, Rigid3d>>& img_specs) {
  Reconstruction rec;
  Camera cam = MakePinholeCamera(1, /*has_prior=*/true);
  rec.AddCameraWithTrivialRig(cam);
  for (const auto& [image_id, pose] : img_specs) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    rec.AddImageWithTrivialFrame(std::move(image), pose);
  }
  return rec;
}

// Both camera centers at the world origin => rays from same point are
// parallel => triangulation angle = 0 => track dropped.
TEST(TrackFilter, FilterTriAngle_ParallelRaysDropped) {
  CorrespondenceGraph view_graph;
  std::unordered_map<point3D_t, Point3D> tracks;

  // Two cameras both at origin, identity orientation => same projection center.
  Reconstruction rec = MakeRecImagesOnly({{1, Rigid3d()}, {2, Rigid3d()}});

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);
  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched =
      FilterTrackTriangulationAngle(view_graph, rec, tracks, /*min_angle=*/1.0);
  EXPECT_EQ(touched, 1);
  EXPECT_EQ(tracks.at(1).track.Length(), 0u);
}

// Two cameras 90deg apart looking at the same point => triangulation angle
// is large (45deg between the two view rays from the point) => track kept.
TEST(TrackFilter, FilterTriAngle_PerpendicularRaysKept) {
  CorrespondenceGraph view_graph;
  std::unordered_map<point3D_t, Point3D> tracks;

  Rigid3d cam1_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(5.0, 0.0, 0.0));
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-5.0, 0.0, 0.0));
  Reconstruction rec =
      MakeRecImagesOnly({{1, cam1_from_world}, {2, cam2_from_world}});

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);
  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched =
      FilterTrackTriangulationAngle(view_graph, rec, tracks, /*min_angle=*/1.0);
  EXPECT_EQ(touched, 0);
  EXPECT_EQ(tracks.at(1).track.Length(), 2u);
}

// Construct a known triangulation angle theta and sweep ``min_angle`` just
// below / just above it.
TEST(TrackFilter, FilterTriAngle_ThreshSweep) {
  CorrespondenceGraph view_graph;

  const double d = 1.0;
  const double h = 10.0;
  const double theta_deg = RadToDeg(2.0 * std::atan2(d, h));

  Rigid3d cam1_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(d, 0.0, 0.0));
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-d, 0.0, 0.0));
  Reconstruction rec =
      MakeRecImagesOnly({{1, cam1_from_world}, {2, cam2_from_world}});

  Eigen::Vector3d xyz(0.0, 0.0, h);

  // min_angle below theta => kept.
  {
    std::unordered_map<point3D_t, Point3D> tracks_kept;
    tracks_kept.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));
    const int touched = FilterTrackTriangulationAngle(
        view_graph, rec, tracks_kept, theta_deg - 0.1);
    EXPECT_EQ(touched, 0);
    EXPECT_EQ(tracks_kept.at(1).track.Length(), 2u);
  }

  // min_angle above theta => dropped.
  {
    std::unordered_map<point3D_t, Point3D> tracks_dropped;
    tracks_dropped.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));
    const int touched = FilterTrackTriangulationAngle(
        view_graph, rec, tracks_dropped, theta_deg + 0.1);
    EXPECT_EQ(touched, 1);
    EXPECT_EQ(tracks_dropped.at(1).track.Length(), 0u);
  }
}

}  // namespace
}  // namespace colmap
