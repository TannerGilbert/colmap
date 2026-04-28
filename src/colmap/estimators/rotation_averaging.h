#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refer
// to the paper "Gravity Aligned Rotation Averaging"
namespace colmap {

struct RotationEstimatorOptions {
  // PRNG seed for stochastic methods during rotation averaging.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the rotation averaging is deterministic with
  // the given seed.
  int random_seed = -1;

  // Maximum number of times to run L1 minimization.
  int max_num_l1_iterations = 5;

  // Average step size threshold to terminate the L1 minimization.
  double l1_step_convergence_threshold = 0.001;

  // The number of iterative reweighted least squares iterations to perform.
  int max_num_irls_iterations = 100;

  // Average step size threshold to terminate the IRLS minimization.
  double irls_step_convergence_threshold = 0.001;

  // Gravity direction.
  Eigen::Vector3d gravity_dir = Eigen::Vector3d(0, 1, 0);

  // The point where the Huber-like cost function switches from L1 to L2.
  double irls_loss_parameter_sigma = 5.0;  // in degrees

  enum WeightType {
    // Geman-McClure weight from "Efficient and robust large-scale rotation
    // averaging" (Chatterjee et al., 2013)
    GEMAN_MCCLURE,
    // Half norm from "Robust Relative Rotation Averaging"
    // (Chatterjee et al., 2017)
    HALF_NORM,
  } weight_type = GEMAN_MCCLURE;

  // Flag to skip maximum spanning tree initialization.
  bool skip_initialization = false;

  // Flag to use gravity priors for rotation averaging.
  bool use_gravity = false;

  // Flag to use stratified solving for mixed gravity systems.
  // If true and use_gravity is true, first solves the 1-DOF system with
  // gravity-only pairs, then solves the full 3-DOF system.
  bool use_stratified = true;

  // If true, only consider frames with existing poses when computing
  // connected components. Set to true for refinement passes.
  bool filter_unregistered = false;

  // If > 0, filter image pairs with rotation error exceeding this threshold
  // after solving, then recompute active set.
  double max_rotation_error_deg = 10.0;

  // Optional extensions (default OFF; disabled = baseline RA).

  // If true, drop pairs whose loop-closure inlier count exceeds non-LC
  // inliers — prevents LC-contaminated pairs from breaking RA. Consumed by
  // BuildPairConstraints in rotation_averaging_impl.cc when a
  // CorrespondenceGraph& is plumbed through.
  bool skip_risky_LC_pairs = false;

  // If true, switch from L1 + IRLS to a Ceres-based solver with differential
  // loss functions (Huber for tracking pairs, Cauchy for LC pairs). Mutually
  // exclusive with use_gravity. Also gates the MST initializer's LC-edge
  // penalty: when true, ``ComputeMaximumPoseGraphSpanningTree`` subtracts
  // kLCPenalty=1e9 from LC-dominated edges so the tree routes through
  // tracking pairs first.
  bool use_video_constraints = false;

  // Loss scales for the video-aware Ceres solver.
  double video_tracking_huber_scale = 0.1;  // ~5.7 degrees
  double video_lc_cauchy_scale = 0.05;      // ~2.8 degrees
};

// High-level interface for rotation averaging.
// Combines problem setup and solving into a single call.
// TODO: Refactor this class into free functions (e.g., EstimateGlobalRotations)
// since it holds no state other than options.
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views.
  // Solves rotation averaging and registers frames with computed poses.
  // active_image_ids defines which images to include.
  // ``final_weights`` (out, optional): per-pair IRLS weight from the last
  // successful iteration. Populated only when SolveIRLS runs.
  // ``correspondence_graph`` (in, optional): Optional CorrespondenceGraph
  // carrying per-pair ImagePair.{inliers, are_lc} used by the LC-aware
  // paths below. Required when ``skip_risky_LC_pairs=true``; nullptr
  // otherwise.
  // Returns true on successful estimation.
  bool EstimateRotations(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction,
      std::unordered_map<image_pair_t, double>* final_weights = nullptr,
      const class CorrespondenceGraph* correspondence_graph = nullptr);

 private:
  // Maybe solves 1-DOF rotation averaging on the gravity-aligned subset.
  // This is the first phase of stratified solving for mixed gravity systems.
  bool MaybeSolveGravityAlignedSubset(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction);

  // Core rotation averaging solver.
  bool SolveRotationAveraging(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction,
      std::unordered_map<image_pair_t, double>* final_weights = nullptr,
      const class CorrespondenceGraph* correspondence_graph = nullptr);

  // Initializes rotations from maximum spanning tree.
  // ``correspondence_graph`` (in, optional): when non-null and
  // ``options_.use_video_constraints`` is true, MST construction
  // penalises LC-dominated edges.
  void InitializeFromMaximumSpanningTree(
      const PoseGraph& pose_graph,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction,
      const class CorrespondenceGraph* correspondence_graph = nullptr);

  const RotationEstimatorOptions options_;
};

// Initialize rig rotations by averaging per-image rotations.
// Estimates cam_from_rig for cameras with unknown calibration,
// then computes rig_from_world for each frame.
bool InitializeRigRotationsFromImages(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    Reconstruction& reconstruction);

// High-level rotation averaging solver that handles rig expansion.
// For cameras with unknown cam_from_rig, first estimates their orientations
// independently using an expanded reconstruction, then initializes the
// cam_from_rig and runs rotation averaging on the original reconstruction.
// ``final_weights`` (out, optional): per-pair IRLS weight from the last
// successful iteration of the FINAL solve (if rig expansion runs, only the
// final solve's weights are returned).
// ``correspondence_graph`` (in, optional): Optional CorrespondenceGraph
// carrying per-pair ImagePair.{inliers, are_lc} used by the LC-aware
// paths below. Required when ``skip_risky_LC_pairs=true``.
bool RunRotationAveraging(
    const RotationEstimatorOptions& options,
    PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    std::unordered_map<image_pair_t, double>* final_weights = nullptr,
    const class CorrespondenceGraph* correspondence_graph = nullptr);

// Compute the maximum spanning tree of ``pose_graph`` over ``image_ids``,
// weighted by ``edge.num_matches``. Returns the root image_id and populates
// ``parents``.
//
// When ``prioritize_tracking=true`` and ``correspondence_graph`` is non-null,
// LC-dominated edges (where ``are_lc`` true count > non-LC inlier count) have
// ``kLCPenalty=1e9`` subtracted from their weight, so the MST routes around
// them. Vanilla colmap behaviour is recovered with
// ``prioritize_tracking=false`` (or a null correspondence_graph).
//
// Exposed in the public header to support unit tests of the LC-penalty
// branch.
image_t ComputeMaximumPoseGraphSpanningTree(
    const PoseGraph& pose_graph,
    const std::unordered_set<image_t>& image_ids,
    std::unordered_map<image_t, image_t>& parents,
    bool prioritize_tracking,
    const class CorrespondenceGraph* correspondence_graph);

}  // namespace colmap
