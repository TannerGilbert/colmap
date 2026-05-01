#include "colmap/sfm/image_pair_inliers.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/models.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TwoViewSetup {
  Reconstruction reconstruction;
  CorrespondenceGraph corr_graph;
  image_t image_id1 = 1;
  image_t image_id2 = 2;
  camera_t camera_id = 1;
  Rigid3d cam2_from_cam1;
};

TwoViewSetup MakeTwoView(const std::vector<Eigen::Vector3d>& cam1_points,
                         const Rigid3d& cam2_from_cam1) {
  TwoViewSetup setup;
  setup.cam2_from_cam1 = cam2_from_cam1;

  Camera cam = Camera::CreateFromModelId(setup.camera_id,
                                         CameraModelId::kSimplePinhole,
                                         /*focal_length=*/1.0,
                                         /*width=*/100,
                                         /*height=*/100);
  setup.reconstruction.AddCamera(std::move(cam));

  const rig_t rig_id = 1;
  Rig rig;
  rig.SetRigId(rig_id);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, setup.camera_id));
  setup.reconstruction.AddRig(std::move(rig));

  Image img1;
  img1.SetImageId(setup.image_id1);
  img1.SetCameraId(setup.camera_id);
  img1.SetFrameId(1);

  Image img2;
  img2.SetImageId(setup.image_id2);
  img2.SetCameraId(setup.camera_id);
  img2.SetFrameId(2);

  for (const auto& p1 : cam1_points) {
    const Eigen::Vector3d p2 = cam2_from_cam1 * p1;
    img1.features_undist.push_back(p1.normalized());
    img2.features_undist.push_back(p2.normalized());
    img1.features.emplace_back(p1.x() / p1.z(), p1.y() / p1.z());
    img2.features.emplace_back(p2.x() / p2.z(), p2.y() / p2.z());
  }

  Frame frame1;
  frame1.SetFrameId(1);
  frame1.SetRigId(rig_id);
  frame1.SetRigFromWorld(Rigid3d());
  frame1.AddDataId(img1.DataId());
  setup.reconstruction.AddFrame(std::move(frame1));

  Frame frame2;
  frame2.SetFrameId(2);
  frame2.SetRigId(rig_id);
  frame2.SetRigFromWorld(cam2_from_cam1);
  frame2.AddDataId(img2.DataId());
  setup.reconstruction.AddFrame(std::move(frame2));

  setup.reconstruction.AddImage(std::move(img1));
  setup.reconstruction.AddImage(std::move(img2));

  CorrespondenceGraph::ImagePair image_pair(setup.image_id1, setup.image_id2);
  image_pair.is_valid = true;
  image_pair.two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  image_pair.two_view_geometry.cam2_from_cam1 = cam2_from_cam1;
  image_pair.two_view_geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  const int num_points = static_cast<int>(cam1_points.size());
  Eigen::MatrixXi matches(num_points, 2);
  for (int idx = 0; idx < num_points; ++idx) {
    matches(idx, 0) = idx;
    matches(idx, 1) = idx;
  }
  image_pair.matches = std::move(matches);

  const image_pair_t pair_id =
      ImagePairToPairId(setup.image_id1, setup.image_id2);
  setup.corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));
  return setup;
}

const CorrespondenceGraph::ImagePair& GetPair(const TwoViewSetup& setup) {
  const image_pair_t pair_id =
      ImagePairToPairId(setup.image_id1, setup.image_id2);
  return setup.corr_graph.ImagePairsMap().at(pair_id);
}

Rigid3d DefaultPose() {
  return Rigid3d(Eigen::Quaterniond::Identity(),
                 Eigen::Vector3d(1.0, 0.0, 0.0));
}

InlierThresholdOptions DefaultOptions() {
  InlierThresholdOptions options;
  options.max_epipolar_error_E = 1e-2;
  options.min_angle_from_epipole = 3.0;
  return options;
}

TEST(ImagePairInlierCount, AllInliersPassFourGates) {
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 5.0},
      {0.0, 0.5, 5.0},
      {0.0, -0.5, 5.0},
      {-0.3, 0.4, 5.0},
      {-0.4, -0.3, 5.0},
  };
  auto setup = MakeTwoView(points, DefaultPose());
  ImagePairsInlierCount(
      setup.corr_graph, setup.reconstruction, DefaultOptions(), true);
  EXPECT_EQ(GetPair(setup).inliers.size(), points.size());
}

TEST(ImagePairInlierCount, EpipolarGateDrops) {
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 5.0},
      {0.0, 0.5, 5.0},
      {0.0, -0.5, 5.0},
      {-0.3, 0.4, 5.0},
      {-0.4, -0.3, 5.0},
      {0.2, 0.2, 5.0},
  };
  auto setup = MakeTwoView(points, DefaultPose());

  auto& image2 = setup.reconstruction.Image(setup.image_id2);
  for (const int idx : {0, 2, 4}) {
    image2.features_undist[idx] += Eigen::Vector3d(0.0, 0.5, 0.0);
    image2.features_undist[idx].normalize();
  }
  ImagePairsInlierCount(
      setup.corr_graph, setup.reconstruction, DefaultOptions(), true);
  const auto& pair = GetPair(setup);
  EXPECT_EQ(pair.inliers.size(), 3u);
  for (const int idx : pair.inliers) {
    EXPECT_TRUE(idx == 1 || idx == 3 || idx == 5) << "idx=" << idx;
  }
}

TEST(ImagePairInlierCount, CheiralityGateDrops) {
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 5.0},
      {0.0, 0.5, 5.0},
  };
  auto setup = MakeTwoView(points, DefaultPose());

  auto& image1 = setup.reconstruction.Image(setup.image_id1);
  auto& image2 = setup.reconstruction.Image(setup.image_id2);
  image1.features_undist[0] = -image1.features_undist[0];
  image2.features_undist[0] = -image2.features_undist[0];

  ImagePairsInlierCount(
      setup.corr_graph, setup.reconstruction, DefaultOptions(), true);
  const auto& pair = GetPair(setup);
  EXPECT_EQ(pair.inliers.size(), 1u);
  EXPECT_EQ(pair.inliers.front(), 1);
}

TEST(ImagePairInlierCount, EpipoleGateDrops) {
  const std::vector<Eigen::Vector3d> points = {
      {100.0, 0.0, 1.0},
      {0.0, 0.5, 5.0},
  };
  auto setup = MakeTwoView(points, DefaultPose());

  InlierThresholdOptions options = DefaultOptions();
  options.min_angle_from_epipole = 30.0;

  ImagePairsInlierCount(setup.corr_graph, setup.reconstruction, options, true);
  const auto& pair = GetPair(setup);
  EXPECT_EQ(pair.inliers.size(), 1u);
  EXPECT_EQ(pair.inliers.front(), 1);
}

}  // namespace
}  // namespace colmap
