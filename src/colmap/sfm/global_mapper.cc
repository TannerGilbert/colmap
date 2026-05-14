#include "colmap/sfm/global_mapper.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/sfm/track_establishment.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace colmap {
namespace {

// Parses a CSV of IMU-derived rotation edges and adds them to the pose
// graph. If an edge already exists for the same image pair (visual match
// from the correspondence graph), the rotation is replaced with the IMU
// estimate (more accurate over short baselines) and num_matches is set to
// max(existing, weight) so the rotation-averaging spanning tree continues
// to favour it. CSV columns: image_id1, image_id2, qw, qx, qy, qz, tx, ty,
// tz, weight. Translation is ignored by rotation averaging but parsed so
// the format stays a strict superset of a generic edge file.
size_t LoadImuEdgesIntoPoseGraph(const std::string& path,
                                 PoseGraph& pose_graph) {
  std::ifstream fs(path);
  THROW_CHECK(fs.is_open()) << "Cannot open IMU edges file: " << path;

  std::string line;
  // Skip header.
  if (!std::getline(fs, line)) {
    return 0;
  }

  size_t n_added = 0;
  size_t n_merged = 0;
  size_t row = 1;
  while (std::getline(fs, line)) {
    ++row;
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string tok;
    auto next = [&]() -> std::string {
      THROW_CHECK(std::getline(ss, tok, ','))
          << "Malformed IMU edges CSV at row " << row;
      return tok;
    };

    const auto id1 = static_cast<image_t>(std::stoll(next()));
    const auto id2 = static_cast<image_t>(std::stoll(next()));
    const double qw = std::stod(next());
    const double qx = std::stod(next());
    const double qy = std::stod(next());
    const double qz = std::stod(next());
    const double tx = std::stod(next());
    const double ty = std::stod(next());
    const double tz = std::stod(next());
    const int weight = static_cast<int>(std::stod(next()));

    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    const Rigid3d cam2_from_cam1(q, Eigen::Vector3d(tx, ty, tz));

    PoseGraph::Edge edge(cam2_from_cam1);
    edge.num_matches = weight;

    if (pose_graph.HasEdge(id1, id2)) {
      const auto& [existing, swapped] = pose_graph.EdgeRef(id1, id2);
      edge.num_matches = std::max(existing.num_matches, weight);

      // Log when the visual rotation we're about to overwrite differs
      // significantly from the IMU rotation. The existing edge stores
      // its rotation in canonical (smaller-id-first) order; bring the
      // new IMU rotation into the same convention before comparing.
      const Rigid3d imu_canonical =
          (id1 > id2) ? Inverse(cam2_from_cam1) : cam2_from_cam1;
      const double disagreement_deg =
          RadToDeg(existing.cam2_from_cam1.rotation().angularDistance(
              imu_canonical.rotation()));
      constexpr double kMergeDisagreementLogThresholdDeg = 5.0;
      if (disagreement_deg > kMergeDisagreementLogThresholdDeg) {
        LOG(INFO) << "  IMU vs visual rotation disagreement on merge: "
                  << "image_pair (" << id1 << ", " << id2
                  << ")  Δ=" << disagreement_deg
                  << " deg  (visual num_matches=" << existing.num_matches
                  << ")";
      }

      pose_graph.UpdateEdge(id1, id2, std::move(edge));
      ++n_merged;
    } else {
      pose_graph.AddEdge(id1, id2, std::move(edge));
      ++n_added;
    }
  }

  LOG(INFO) << "IMU pose-graph edges: " << n_added << " added, " << n_merged
            << " merged with existing visual edges";
  return n_added + n_merged;
}

bool RunBundleAdjustment(const BundleAdjustmentOptions& options,
                         Reconstruction& reconstruction,
                         bool use_prior_position = false,
                         bool use_robust_loss_on_prior_position = false,
                         double prior_position_loss_scale = 7.815,
                         const std::vector<PosePrior>& pose_priors = {}) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }

  const bool use_priors =
      use_prior_position && ba_config.NumImages() > 2 && !pose_priors.empty();
  std::unique_ptr<BundleAdjuster> ba;
  if (use_priors) {
    PosePriorBundleAdjustmentOptions prior_options;
    if (use_robust_loss_on_prior_position) {
      prior_options.ceres->prior_position_loss.type =
          CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;
    }
    prior_options.ceres->prior_position_loss.scale = prior_position_loss_scale;
    ba = CreatePosePriorBundleAdjuster(
        options, prior_options, ba_config, pose_priors, reconstruction);
  } else {
    ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
    ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);
  }

  return ba->Solve()->IsSolutionUsable();
}

}  // namespace

RotationEstimatorOptions GlobalMapperOptions::RotationAveraging() const {
  RotationEstimatorOptions opts = rotation_averaging;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalPositionerOptions GlobalMapperOptions::GlobalPositioning() const {
  GlobalPositionerOptions opts = global_positioning;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  opts.solver_options.num_threads = num_threads;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
    opts.use_parameter_block_ordering = false;
  }
  return opts;
}

BundleAdjustmentOptions GlobalMapperOptions::BundleAdjustment() const {
  BundleAdjustmentOptions opts = bundle_adjustment;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (opts.ceres) {
    opts.ceres->solver_options.num_threads = num_threads;
  }
  return opts;
}

IncrementalTriangulator::Options GlobalMapperOptions::Retriangulation() const {
  IncrementalTriangulator::Options opts = retriangulation;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalMapper::GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Read pose priors from the database cache.
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!RunRotationAveraging(custom_options,
                            *pose_graph_,
                            *reconstruction_,
                            pose_priors,
                            nullptr,
                            database_cache_->CorrespondenceGraph().get())) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!RunRotationAveraging(custom_options,
                            *pose_graph_,
                            *reconstruction_,
                            pose_priors,
                            nullptr,
                            database_cache_->CorrespondenceGraph().get())) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const GlobalMapperOptions& options) {
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  // Build keypoints map from registered images.
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  for (const auto image_id : reconstruction_->RegImageIds()) {
    const auto& image = reconstruction_->Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const auto& point2D : image.Points2D()) {
      points.push_back(point2D.xy);
    }
    image_id_to_keypoints.emplace(image_id, std::move(points));
  }

  TrackEstablishmentOptions to;
  to.intra_image_consistency_threshold =
      options.track_intra_image_consistency_threshold;
  to.min_num_views_per_track = options.track_min_num_views_per_track;
  // LC second pass appends observations after the union-find pass, so it needs
  // the full candidate set. Without LC, preserve native per-view limiting.
  to.required_tracks_per_view = options.track_lc_second_pass
                                    ? std::numeric_limits<int>::max()
                                    : options.track_required_tracks_per_view;

  std::vector<image_pair_t> valid_pair_ids;
  valid_pair_ids.reserve(pose_graph_->NumEdges());
  for (const auto& [pair_id, edge] : pose_graph_->ValidEdges()) {
    valid_pair_ids.push_back(pair_id);
  }

  MatchPredicate ignore_match;
  if (options.track_lc_second_pass) {
    ignore_match = MakeLoopClosureMatchPredicate(
        valid_pair_ids, *database_cache_->CorrespondenceGraph());
  }
  auto selected =
      EstablishTracksFromCorrGraph(valid_pair_ids,
                                   *database_cache_->CorrespondenceGraph(),
                                   image_id_to_keypoints,
                                   to,
                                   ignore_match);
  if (options.track_lc_second_pass) {
    AppendLoopClosureObservations(
        valid_pair_ids, *database_cache_->CorrespondenceGraph(), selected);
  }
  for (auto& [point3D_id, point3D] : selected) {
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));
  }
  LOG(INFO) << "Track establishment: " << reconstruction_->NumPoints3D()
            << " tracks added to reconstruction";
}

bool GlobalMapper::GlobalPositioning(const GlobalPositionerOptions& options,
                                     double max_angular_reproj_error_deg,
                                     double max_normalized_reproj_error,
                                     double min_tri_angle_deg,
                                     bool use_prior_position) {
  if (!RunGlobalPositioning(options, *pose_graph_, *reconstruction_)) {
    return false;
  }

  // Filter tracks based on the estimation
  ObservationManager obs_manager(*reconstruction_);

  // First pass: use relaxed threshold (2x) for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * max_angular_reproj_error_deg,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::ANGULAR);

  // Second pass: apply strict threshold for cameras with prior focal length.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  std::vector<std::pair<image_t, point2D_t>> obs_to_delete;
  for (const auto point3D_id : reconstruction_->Point3DIds()) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const auto& point3D = reconstruction_->Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& image = reconstruction_->Image(track_el.image_id);
      const auto& camera = *image.CameraPtr();
      if (!camera.has_prior_focal_length) {
        continue;
      }
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const double error = CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (error > max_angular_error_rad) {
        obs_to_delete.emplace_back(track_el.image_id, track_el.point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : obs_to_delete) {
    if (reconstruction_->Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
      obs_manager.DeleteObservation(image_id, point2D_idx);
    }
  }

  // Filter tracks based on triangulation angle and reprojection error
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  // Normalize the structure for numerical stability if no prior position is
  // provided.
  if (!use_prior_position) {
    reconstruction_->Normalize();
  }

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjustmentOptions& options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage,
    bool use_prior_position,
    bool use_robust_loss_on_prior_position,
    double prior_position_loss_scale) {
  const std::vector<PosePrior> empty_priors;
  const std::vector<PosePrior>& pose_priors =
      use_prior_position ? database_cache_->PosePriors() : empty_priors;
  for (int ite = 0; ite < num_iterations; ite++) {
    // Optional fixed-rotation stage: optimize positions only
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      if (!RunBundleAdjustment(opts_position_only,
                               *reconstruction_,
                               use_prior_position,
                               use_robust_loss_on_prior_position,
                               prior_position_loss_scale,
                               pose_priors)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << num_iterations << ", fixed-rotation stage finished";
    }

    // Joint optimization stage: default BA
    if (!skip_joint_optimization_stage) {
      if (!RunBundleAdjustment(options,
                               *reconstruction_,
                               use_prior_position,
                               use_robust_loss_on_prior_position,
                               prior_position_loss_scale,
                               pose_priors)) {
        return false;
      }
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << " finished";

    // Normalize the structure for numerical stability if no prior position is
    // provided.
    if (!use_prior_position) {
      reconstruction_->Normalize();
    }

    // Filter tracks based on the estimation
    // For the filtering, in each round, the criteria for outlier is
    // tightened. If only few tracks are changed, no need to start bundle
    // adjustment right away. Instead, use a more strict criteria to filter
    LOG(INFO) << "Filtering tracks by reprojection ...";

    ObservationManager obs_manager(*reconstruction_);
    bool status = true;
    size_t filtered_num = 0;
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_normalized_reproj_error,
          reconstruction_->Point3DIds(),
          ReprojectionErrorType::NORMALIZED);

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  // Filter tracks based on the estimation
  LOG(INFO) << "Filtering tracks by reprojection ...";
  {
    ObservationManager obs_manager(*reconstruction_);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        reconstruction_->Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        min_tri_angle_deg, reconstruction_->Point3DIds());
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const IncrementalTriangulator::Options& options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    bool use_prior_position,
    bool use_robust_loss_on_prior_position,
    double prior_position_loss_scale) {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  BundleAdjustmentOptions custom_ba_options = ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations = 50;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
  }

  // Iterative global refinement.
  IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  mapper_options.use_prior_position = use_prior_position;
  mapper_options.use_robust_loss_on_prior_position =
      use_robust_loss_on_prior_position;
  mapper_options.prior_position_loss_scale = prior_position_loss_scale;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   custom_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/
                                   !use_prior_position);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  ObservationManager obs_manager(*reconstruction_);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  const std::vector<PosePrior> empty_priors;
  const std::vector<PosePrior>& pose_priors =
      use_prior_position ? database_cache_->PosePriors() : empty_priors;
  if (!RunBundleAdjustment(ba_options,
                           *reconstruction_,
                           use_prior_position,
                           use_robust_loss_on_prior_position,
                           prior_position_loss_scale,
                           pose_priors)) {
    return false;
  }

  // Normalize the structure for numerical stability if no prior position is
  // provided.
  if (!use_prior_position) {
    reconstruction_->Normalize();
  }

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());

  return true;
}

bool GlobalMapper::Solve(const GlobalMapperOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Inject IMU-derived rotation edges before rotation averaging runs, so
  // both RA passes (and gravity refinement, if enabled) see them.
  if (!options.imu_edges_path.empty()) {
    const size_t n =
        LoadImuEdgesIntoPoseGraph(options.imu_edges_path, *pose_graph_);
    LOG(INFO) << "Loaded " << n << " IMU rotation edges into pose graph";
  }

  // Run rotation averaging
  if (!options.skip_rotation_averaging) {
    LOG_HEADING1("Running rotation averaging");
    Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(options.RotationAveraging())) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!options.skip_track_establishment) {
    LOG_HEADING1("Running track establishment");
    Timer run_timer;
    run_timer.Start();
    EstablishTracks(options);
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (!options.skip_global_positioning) {
    LOG_HEADING1("Running global positioning");
    Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(options.GlobalPositioning(),
                           options.max_angular_reproj_error_deg,
                           options.max_normalized_reproj_error,
                           options.min_tri_angle_deg,
                           options.use_prior_position)) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Bundle adjustment
  if (!options.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment");
    if (options.use_prior_position) {
      LOG(INFO) << "Pose-prior BA active: "
                << database_cache_->PosePriors().size()
                << " priors will be applied (robust_loss="
                << options.use_robust_loss_on_prior_position
                << ", loss_scale=" << options.prior_position_loss_scale << ").";
    }
    Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(options.BundleAdjustment(),
                                   options.max_normalized_reproj_error,
                                   options.min_tri_angle_deg,
                                   options.ba_num_iterations,
                                   options.ba_skip_fixed_rotation_stage,
                                   options.ba_skip_joint_optimization_stage,
                                   options.use_prior_position,
                                   options.use_robust_loss_on_prior_position,
                                   options.prior_position_loss_scale)) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!options.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(
            options.Retriangulation(),
            options.BundleAdjustment(),
            options.max_normalized_reproj_error,
            options.min_tri_angle_deg,
            options.use_prior_position,
            options.use_robust_loss_on_prior_position,
            options.prior_position_loss_scale)) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  return true;
}

}  // namespace colmap
