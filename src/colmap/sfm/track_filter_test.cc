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

#include "colmap/sfm/track_filter.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
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
  Camera camera = Camera::CreateFromModelName(
      camera_id, "PINHOLE", /*focal_length=*/100.0,
      /*width=*/200, /*height=*/200);
  camera.has_prior_focal_length = has_prior;
  return camera;
}

// Build an Image at world-position ``center`` looking down +Z (rotation =
// identity for cam_from_world means world is identical to camera frame). Set
// camera_id and reserve enough features_undist slots.
Image MakeImage(image_t image_id,
                camera_t camera_id,
                const Rigid3d& cam_from_world,
                size_t num_features) {
  Image image;
  image.SetImageId(image_id);
  image.SetCameraId(camera_id);
  image.cam_from_world = cam_from_world;
  image.features_undist.assign(num_features, Eigen::Vector3d::UnitZ());
  return image;
}

// Build a Point3D with track over (image_id, point2D_idx) entries.
Point3D MakePoint3D(const Eigen::Vector3d& xyz,
                    const std::vector<std::pair<image_t, point2D_t>>& obs) {
  Point3D point3D;
  point3D.xyz = xyz;
  point3D.is_initialized = true;
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
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  cameras.emplace(1, MakePinholeCamera(1, /*has_prior=*/true));

  // Camera 1 at origin, identity rotation.
  Image img1 = MakeImage(1, 1, Rigid3d(), 1);
  // Camera 2 at (1, 0, 0), identity rotation.
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-1.0, 0.0, 0.0));
  Image img2 = MakeImage(2, 1, cam2_from_world, 1);

  // Point at (0, 0, 5) in world.
  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  // Set features_undist to the normalized rays of the projection in each cam.
  img1.features_undist[0] = (img1.cam_from_world * xyz).normalized();
  img2.features_undist[0] = (img2.cam_from_world * xyz).normalized();

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched = FilterTracksByAngle(
      view_graph, cameras, images, tracks, /*max_angle_error=*/1.0);
  EXPECT_EQ(touched, 0);
  EXPECT_EQ(tracks.at(1).track.Length(), 2u);
}

// Rays misaligned by 5deg from the back-projection => above 1deg threshold
// for calibrated cameras => both observations dropped.
TEST(TrackFilter, FilterByAngle_MisalignedRaysDropped) {
  CorrespondenceGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  cameras.emplace(1, MakePinholeCamera(1, /*has_prior=*/true));

  Image img1 = MakeImage(1, 1, Rigid3d(), 1);
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-1.0, 0.0, 0.0));
  Image img2 = MakeImage(2, 1, cam2_from_world, 1);

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  // Apply a 5deg rotation (around Y) to the back-projected ray to emulate
  // observed feature_undist that is 5deg off the ground-truth ray.
  const double angle_rad = DegToRad(5.0);
  Eigen::AngleAxisd tilt(angle_rad, Eigen::Vector3d::UnitY());
  img1.features_undist[0] = (tilt * (img1.cam_from_world * xyz)).normalized();
  img2.features_undist[0] = (tilt * (img2.cam_from_world * xyz)).normalized();

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched = FilterTracksByAngle(
      view_graph, cameras, images, tracks, /*max_angle_error=*/1.0);
  EXPECT_EQ(touched, 1);
  EXPECT_EQ(tracks.at(1).track.Length(), 0u);
}

// Same 5deg misalignment, but camera is uncalibrated => threshold doubles to
// 10deg => both observations kept.
TEST(TrackFilter, FilterByAngle_UncalibratedCameraDoubleThreshold) {
  CorrespondenceGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  // Uncalibrated camera => threshold becomes 2 * max_angle_error.
  cameras.emplace(1, MakePinholeCamera(1, /*has_prior=*/false));

  Image img1 = MakeImage(1, 1, Rigid3d(), 1);
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-1.0, 0.0, 0.0));
  Image img2 = MakeImage(2, 1, cam2_from_world, 1);

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  // 5deg misalignment is above 1deg (calibrated) but below 2deg ?? No: above
  // 2deg too. Threshold is 2 * max_angle_error = 2 * 3deg = 6deg in this test.
  // We use max_angle_error=3.0 so calibrated-thres=3deg (drop 5deg) but
  // uncalib-thres=6deg (keep 5deg).
  const double angle_rad = DegToRad(5.0);
  Eigen::AngleAxisd tilt(angle_rad, Eigen::Vector3d::UnitY());
  img1.features_undist[0] = (tilt * (img1.cam_from_world * xyz)).normalized();
  img2.features_undist[0] = (tilt * (img2.cam_from_world * xyz)).normalized();

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched = FilterTracksByAngle(
      view_graph, cameras, images, tracks, /*max_angle_error=*/3.0);
  EXPECT_EQ(touched, 0);
  EXPECT_EQ(tracks.at(1).track.Length(), 2u);

  // Sanity: same setup, but now flip the camera to calibrated => 5deg drops.
  std::unordered_map<camera_t, Camera> cameras_calib;
  cameras_calib.emplace(1, MakePinholeCamera(1, /*has_prior=*/true));
  std::unordered_map<point3D_t, Point3D> tracks_calib;
  tracks_calib.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));
  const int touched_calib = FilterTracksByAngle(
      view_graph, cameras_calib, images, tracks_calib,
      /*max_angle_error=*/3.0);
  EXPECT_EQ(touched_calib, 1);
  EXPECT_EQ(tracks_calib.at(1).track.Length(), 0u);
}

// Point behind camera (cam-frame z < EPS) => observation is skipped (the
// implementation does ``continue`` so it is dropped from the new track).
// Verify only the behind-camera obs is dropped; valid one stays.
TEST(TrackFilter, FilterByAngle_PointBehindCameraSkipped) {
  CorrespondenceGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  cameras.emplace(1, MakePinholeCamera(1, /*has_prior=*/true));

  // Image 1 looks +Z (point at z=5 is in front).
  Image img1 = MakeImage(1, 1, Rigid3d(), 1);
  // Image 2 looks -Z by rotating 180deg around Y; same world point lands at
  // cam-frame z = -5 (behind camera).
  Eigen::Quaterniond q_flip(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
  Rigid3d cam2_from_world(q_flip, Eigen::Vector3d::Zero());
  Image img2 = MakeImage(2, 1, cam2_from_world, 1);

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  img1.features_undist[0] = (img1.cam_from_world * xyz).normalized();
  // For img2, give a perfectly aligned feature_undist regardless: the
  // ``z < EPS`` early continue kicks in before the angle check.
  img2.features_undist[0] = Eigen::Vector3d::UnitZ();

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched = FilterTracksByAngle(
      view_graph, cameras, images, tracks, /*max_angle_error=*/1.0);
  EXPECT_EQ(touched, 1);
  // Only image 1's observation survived.
  ASSERT_EQ(tracks.at(1).track.Length(), 1u);
  EXPECT_EQ(tracks.at(1).track.Element(0).image_id, 1u);
}

// === FilterTrackTriangulationAngle tests ===

// Both camera centers at the world origin => rays from same point are
// parallel => triangulation angle = 0 => track dropped.
TEST(TrackFilter, FilterTriAngle_ParallelRaysDropped) {
  CorrespondenceGraph view_graph;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  // Two cameras both at origin, identity orientation => same projection center.
  Image img1 = MakeImage(1, 1, Rigid3d(), 1);
  Image img2 = MakeImage(2, 1, Rigid3d(), 1);

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched = FilterTrackTriangulationAngle(
      view_graph, images, tracks, /*min_angle=*/1.0);
  EXPECT_EQ(touched, 1);
  EXPECT_EQ(tracks.at(1).track.Length(), 0u);
}

// Two cameras 90deg apart looking at the same point => triangulation angle
// is large (45deg between the two view rays from the point) => track kept.
TEST(TrackFilter, FilterTriAngle_PerpendicularRaysKept) {
  CorrespondenceGraph view_graph;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  // Cam1 center at (-5, 0, 0), Cam2 center at (5, 0, 0). Point at (0, 0, 5).
  // Rays from point: (5, 0, -5)/|.| and (-5, 0, -5)/|.|. Dot product = 0
  // i.e. 90deg triangulation angle => well above 1deg threshold.
  // We only need the projection center, so cam_from_world is just t = -R*c.
  Rigid3d cam1_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(5.0, 0.0, 0.0));
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-5.0, 0.0, 0.0));

  Image img1 = MakeImage(1, 1, cam1_from_world, 1);
  Image img2 = MakeImage(2, 1, cam2_from_world, 1);

  Eigen::Vector3d xyz(0.0, 0.0, 5.0);

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  tracks.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));

  const int touched = FilterTrackTriangulationAngle(
      view_graph, images, tracks, /*min_angle=*/1.0);
  EXPECT_EQ(touched, 0);
  EXPECT_EQ(tracks.at(1).track.Length(), 2u);
}

// Construct a known triangulation angle theta and sweep ``min_angle`` just
// below / just above it.
TEST(TrackFilter, FilterTriAngle_ThreshSweep) {
  CorrespondenceGraph view_graph;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;

  // Place camera centers at C1=(-d, 0, 0), C2=(d, 0, 0); point at P=(0, 0, h).
  // Triangulation angle theta = 2 * atan2(d, h).
  const double d = 1.0;
  const double h = 10.0;
  const double theta_deg = RadToDeg(2.0 * std::atan2(d, h));

  Rigid3d cam1_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(d, 0.0, 0.0));
  Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d(-d, 0.0, 0.0));

  Image img1 = MakeImage(1, 1, cam1_from_world, 1);
  Image img2 = MakeImage(2, 1, cam2_from_world, 1);

  images.emplace(1, std::move(img1));
  images.emplace(2, std::move(img2));

  Eigen::Vector3d xyz(0.0, 0.0, h);

  // min_angle below theta => kept.
  {
    std::unordered_map<point3D_t, Point3D> tracks_kept;
    tracks_kept.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));
    const int touched = FilterTrackTriangulationAngle(
        view_graph, images, tracks_kept, theta_deg - 0.1);
    EXPECT_EQ(touched, 0);
    EXPECT_EQ(tracks_kept.at(1).track.Length(), 2u);
  }

  // min_angle above theta => dropped.
  {
    std::unordered_map<point3D_t, Point3D> tracks_dropped;
    tracks_dropped.emplace(1, MakePoint3D(xyz, {{1, 0}, {2, 0}}));
    const int touched = FilterTrackTriangulationAngle(
        view_graph, images, tracks_dropped, theta_deg + 0.1);
    EXPECT_EQ(touched, 1);
    EXPECT_EQ(tracks_dropped.at(1).track.Length(), 0u);
  }
}

}  // namespace
}  // namespace colmap
