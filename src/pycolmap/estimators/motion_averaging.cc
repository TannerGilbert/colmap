#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/gravity_refinement.h"
#include "colmap/estimators/rotation_averaging.h"

#include "pycolmap/helpers.h"

#include <cmath>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindGlobalPositioner(py::module& m) {
  // ``LossConfig`` is bound by ``BindBundleAdjuster`` (estimators/
  // bundle_adjustment.cc), which runs earlier in ``BindEstimators``.
  auto PyGlobalPositionerOptions =
      py::classh<GlobalPositionerOptions>(m, "GlobalPositionerOptions")
          .def(py::init<>())
          .def_readwrite("generate_random_positions",
                         &GlobalPositionerOptions::generate_random_positions,
                         "Whether to initialize camera positions randomly.")
          .def_readwrite("generate_random_points",
                         &GlobalPositionerOptions::generate_random_points,
                         "Whether to initialize 3D point positions randomly.")
          .def_readwrite("generate_scales",
                         &GlobalPositionerOptions::generate_scales,
                         "Whether to initialize scales to constant 1 or derive "
                         "from positions.")
          .def_readwrite("optimize_positions",
                         &GlobalPositionerOptions::optimize_positions,
                         "Whether to optimize camera positions.")
          .def_readwrite("optimize_points",
                         &GlobalPositionerOptions::optimize_points,
                         "Whether to optimize 3D point positions.")
          .def_readwrite("optimize_scales",
                         &GlobalPositionerOptions::optimize_scales,
                         "Whether to optimize scales.")
          .def_readwrite("use_gpu",
                         &GlobalPositionerOptions::use_gpu,
                         "Whether to use GPU for optimization.")
          .def_readwrite("gpu_index",
                         &GlobalPositionerOptions::gpu_index,
                         "GPU device index (-1 for auto).")
          .def_readwrite("min_num_images_gpu_solver",
                         &GlobalPositionerOptions::min_num_images_gpu_solver,
                         "Minimum number of images to use GPU solver.")
          .def_readwrite("min_num_view_per_track",
                         &GlobalPositionerOptions::min_num_view_per_track,
                         "Minimum number of views per track.")
          .def_readwrite("random_seed",
                         &GlobalPositionerOptions::random_seed,
                         "PRNG seed for random initialization. Default -1 "
                         "(non-deterministic random_device, matches upstream "
                         "colmap4). When -1 the ctor honors a GP_SEED env "
                         "var as a documented escape for byte-identity "
                         "recipes; set explicitly (>=0) to override.")
          .def_readwrite("loss",
                         &GlobalPositionerOptions::loss,
                         "Top-level robust loss applied to the BATA "
                         "direction residual (LossConfig: type, scale, "
                         "weight). Default HUBER@0.1 — was hardcoded in "
                         "upstream colmap GP.")
          .def_readwrite("use_parameter_block_ordering",
                         &GlobalPositionerOptions::use_parameter_block_ordering,
                         "Whether to use custom parameter block ordering.")
          .def_readwrite(
              "apply_uncalibrated_loss_downweight",
              &GlobalPositionerOptions::apply_uncalibrated_loss_downweight,
              "Apply 0.5x ScaledLoss to BATA residuals from cameras whose "
              "focal length lacks an EXIF prior. Default true. Set false "
              "to disable the downweight.")
          .def_property(
              "num_threads",
              [](const GlobalPositionerOptions& self) {
                return self.solver_options.num_threads;
              },
              [](GlobalPositionerOptions& self, int v) {
                self.solver_options.num_threads = v;
              },
              "Ceres solver thread count (-1 = auto).")
          .def_property(
              "max_num_iterations",
              [](const GlobalPositionerOptions& self) {
                return self.solver_options.max_num_iterations;
              },
              [](GlobalPositionerOptions& self, int v) {
                self.solver_options.max_num_iterations = v;
              },
              "Ceres solver max iterations.")
          .def_property(
              "function_tolerance",
              [](const GlobalPositionerOptions& self) {
                return self.solver_options.function_tolerance;
              },
              [](GlobalPositionerOptions& self, double v) {
                self.solver_options.function_tolerance = v;
              },
              "Ceres solver function tolerance.")
          .def_property(
              "gradient_tolerance",
              [](const GlobalPositionerOptions& self) {
                return self.solver_options.gradient_tolerance;
              },
              [](GlobalPositionerOptions& self, double v) {
                self.solver_options.gradient_tolerance = v;
              },
              "Ceres solver gradient tolerance.")
          .def_property(
              "parameter_tolerance",
              [](const GlobalPositionerOptions& self) {
                return self.solver_options.parameter_tolerance;
              },
              [](GlobalPositionerOptions& self, double v) {
                self.solver_options.parameter_tolerance = v;
              },
              "Ceres solver parameter tolerance.")
          // Optional extensions (default OFF — vanilla call = vanilla GP).
          .def_readwrite(
              "use_metric_depth_constraint",
              &GlobalPositionerOptions::use_metric_depth_constraint,
              "If true, each observation contributes a 1-D MetricDepthError "
              "residual on top of the BATA direction residual. Requires "
              "image.depth_prior_validity[idx] populated.")
          .def_readwrite("use_init",
                         &GlobalPositionerOptions::use_init,
                         "If true, skip random init for both camera centers "
                         "and track xyz.")
          .def_readwrite(
              "use_lc_observations",
              &GlobalPositionerOptions::use_lc_observations,
              "If true, AddPoint3DToProblem also iterates "
              "track.lc_elements (loop-closure observations).")
          .def_readwrite(
              "random_init_scale",
              &GlobalPositionerOptions::random_init_scale,
              "Cube size for random init of camera centers / points (linear).")
          .def_readwrite(
              "use_log_scale_for_depth_map_scales",
              &GlobalPositionerOptions::use_log_scale_for_depth_map_scales,
              "If true, dmap_scales_ are log-space and use exp() in "
              "MetricDepthError.")
          .def_readwrite(
              "use_log_residual_for_depth",
              &GlobalPositionerOptions::use_log_residual_for_depth,
              "If true, use log-space residual in MetricDepthError for "
              "points in front of camera.")
          .def_readwrite(
              "zero_residual_behind",
              &GlobalPositionerOptions::zero_residual_behind,
              "If true, set MetricDepthError residual to 0 for points "
              "behind camera.")
          .def_readwrite(
              "smooth_log_linear_transition",
              &GlobalPositionerOptions::smooth_log_linear_transition,
              "If true, C1-blend log<->linear residual at threshold "
              "(use_log_residual_for_depth=true only).")
          .def_readwrite(
              "log_linear_threshold",
              &GlobalPositionerOptions::log_linear_threshold,
              "z-depth threshold for smooth_log_linear_transition.")
          .def_readwrite(
              "scale_prior_stddev",
              &GlobalPositionerOptions::scale_prior_stddev,
              "Per-image scale-prior stddev (linear or log).")
          .def_readwrite(
              "filter_depth_outliers",
              &GlobalPositionerOptions::filter_depth_outliers,
              "If true, run pre-Solve 3-sigma log-space depth-outlier "
              "filter.")
          .def_property(
              "initial_dmap_scales",
              [](const GlobalPositionerOptions& self) -> py::object {
                if (!self.initial_dmap_scales.has_value()) {
                  return py::none();
                }
                py::dict d;
                for (const auto& [image_id, scale] :
                     *self.initial_dmap_scales) {
                  d[py::cast(image_id)] = scale;
                }
                return d;
              },
              [](GlobalPositionerOptions& self, py::object value) {
                if (value.is_none()) {
                  self.initial_dmap_scales.reset();
                  return;
                }
                std::unordered_map<image_t, double> map;
                for (auto item : py::cast<py::dict>(value)) {
                  map[py::cast<image_t>(item.first)] =
                      py::cast<double>(item.second);
                }
                self.initial_dmap_scales = std::move(map);
              },
              "Caller-supplied {image_id: linear_scale} seed for dmap_scales_ "
              "(GP1 -> GP2 handoff). None = use defaults.");

  // 10 per-bucket loss configs. ``LossConfig`` carries
  // (type=LossFunctionType enum, scale, weight). Defaults give
  // unweighted TrivialLoss — equivalent to no override.
  PyGlobalPositionerOptions
      .def_readwrite("loss_normal_geometry",
                     &GlobalPositionerOptions::loss_normal_geometry)
      .def_readwrite("loss_normal_depth",
                     &GlobalPositionerOptions::loss_normal_depth)
      .def_readwrite("loss_lc_geometry",
                     &GlobalPositionerOptions::loss_lc_geometry)
      .def_readwrite("loss_lc_depth", &GlobalPositionerOptions::loss_lc_depth)
      .def_readwrite("loss_normal_geometry_inlier",
                     &GlobalPositionerOptions::loss_normal_geometry_inlier)
      .def_readwrite("loss_normal_depth_inlier",
                     &GlobalPositionerOptions::loss_normal_depth_inlier)
      .def_readwrite("loss_normal_depth_outlier",
                     &GlobalPositionerOptions::loss_normal_depth_outlier)
      .def_readwrite("loss_normal_geometry_trackstart",
                     &GlobalPositionerOptions::loss_normal_geometry_trackstart)
      .def_readwrite("loss_normal_depth_trackstart",
                     &GlobalPositionerOptions::loss_normal_depth_trackstart)
      .def_readwrite("loss_scale_prior",
                     &GlobalPositionerOptions::loss_scale_prior);

  MakeDataclass(PyGlobalPositionerOptions);

  m.def(
      "run_global_positioning",
      [](const GlobalPositionerOptions& options,
         const PoseGraph& pose_graph,
         Reconstruction& reconstruction) {
        GlobalPositioner positioner(options);
        bool success = false;
        {
          py::gil_scoped_release release;
          success = positioner.Solve(pose_graph, reconstruction);
        }
        // Convert dmap_scales_ to linear-space dict for return.
        py::dict dmap_scale_map;
        for (const auto& [image_id, scale] : positioner.GetDmapScales()) {
          const double linear = options.use_log_scale_for_depth_map_scales
                                    ? std::exp(scale)
                                    : scale;
          dmap_scale_map[py::cast(image_id)] = linear;
        }
        py::dict result;
        result["success"] = success;
        result["dmap_scale_map"] = dmap_scale_map;
        return result;
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "Solve global positioning using point-to-camera constraints. Returns "
      "a dict {'success': bool, 'dmap_scale_map': Dict[image_id, float]}. "
      "``reconstruction`` is mutated in place with the optimized poses + "
      "track xyz.");
}

void BindGravityRefiner(py::module& m) {
  auto PyGravityRefinerOptions =
      py::classh<GravityRefinerOptions>(m, "GravityRefinerOptions")
          .def(py::init<>())
          .def_readwrite(
              "max_outlier_ratio",
              &GravityRefinerOptions::max_outlier_ratio,
              "Maximum ratio that gravity should be consistent with.")
          .def_readwrite("max_gravity_error",
                         &GravityRefinerOptions::max_gravity_error,
                         "Maximum allowed angle error in degrees.")
          .def_readwrite("min_num_neighbors",
                         &GravityRefinerOptions::min_num_neighbors,
                         "Minimum neighbors required for refinement.");
  MakeDataclass(PyGravityRefinerOptions);

  m.def(
      "run_gravity_refinement",
      [](const GravityRefinerOptions& options,
         const PoseGraph& pose_graph,
         const Reconstruction& reconstruction,
         std::vector<PosePrior>& pose_priors) {
        py::gil_scoped_release release;
        RunGravityRefinement(options, pose_graph, reconstruction, pose_priors);
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "pose_priors"_a,
      "Refine gravity stored in pose priors using relative rotations from the "
      "pose graph. Modifies pose_priors in-place.");
}

void BindRotationEstimator(py::module& m) {
  using WeightType = RotationEstimatorOptions::WeightType;
  auto PyWeightType = py::enum_<WeightType>(m, "RotationWeightType")
                          .value("GEMAN_MCCLURE", WeightType::GEMAN_MCCLURE)
                          .value("HALF_NORM", WeightType::HALF_NORM);
  AddStringToEnumConstructor(PyWeightType);

  auto PyRotationEstimatorOptions =
      py::classh<RotationEstimatorOptions>(m, "RotationEstimatorOptions")
          .def(py::init<>())
          .def_readwrite("random_seed",
                         &RotationEstimatorOptions::random_seed,
                         "PRNG seed. -1 for non-deterministic, >=0 for "
                         "deterministic.")
          .def_readwrite("max_num_l1_iterations",
                         &RotationEstimatorOptions::max_num_l1_iterations,
                         "Maximum number of L1 minimization iterations.")
          .def_readwrite(
              "l1_step_convergence_threshold",
              &RotationEstimatorOptions::l1_step_convergence_threshold,
              "Average step size threshold to terminate L1 minimization.")
          .def_readwrite("max_num_irls_iterations",
                         &RotationEstimatorOptions::max_num_irls_iterations,
                         "Number of IRLS iterations to perform.")
          .def_readwrite(
              "irls_step_convergence_threshold",
              &RotationEstimatorOptions::irls_step_convergence_threshold,
              "Average step size threshold to terminate IRLS.")
          .def_readwrite("gravity_dir",
                         &RotationEstimatorOptions::gravity_dir,
                         "Gravity direction vector.")
          .def_readwrite(
              "irls_loss_parameter_sigma",
              &RotationEstimatorOptions::irls_loss_parameter_sigma,
              "Point where Huber-like cost switches from L1 to L2 (degrees).")
          .def_readwrite("weight_type",
                         &RotationEstimatorOptions::weight_type,
                         "Weight type for IRLS: GEMAN_MCCLURE or HALF_NORM.")
          .def_readwrite("skip_initialization",
                         &RotationEstimatorOptions::skip_initialization,
                         "Skip maximum spanning tree initialization.")
          .def_readwrite("use_gravity",
                         &RotationEstimatorOptions::use_gravity,
                         "Use gravity priors for rotation averaging.")
          .def_readwrite("use_stratified",
                         &RotationEstimatorOptions::use_stratified,
                         "Use stratified solving for mixed gravity systems.")
          .def_readwrite(
              "filter_unregistered",
              &RotationEstimatorOptions::filter_unregistered,
              "Only consider frames with existing poses for connected "
              "components.")
          .def_readwrite(
              "max_rotation_error_deg",
              &RotationEstimatorOptions::max_rotation_error_deg,
              "Filter pairs with rotation error exceeding this threshold "
              "(degrees).")
          // --- Video / loop-closure extensions ---
          .def_readwrite(
              "skip_risky_LC_pairs",
              &RotationEstimatorOptions::skip_risky_LC_pairs,
              "Drop pairs whose LC inliers exceed non-LC inliers.")
          .def_readwrite(
              "use_video_constraints",
              &RotationEstimatorOptions::use_video_constraints,
              "Use Ceres video-aware solver with differential loss "
              "functions. Mutually exclusive with use_gravity. Also gates "
              "the LC-penalty branch in the MST initializer.")
          .def_readwrite(
              "video_tracking_huber_scale",
              &RotationEstimatorOptions::video_tracking_huber_scale,
              "Huber loss scale for tracking pairs in the video solver.")
          .def_readwrite(
              "video_lc_cauchy_scale",
              &RotationEstimatorOptions::video_lc_cauchy_scale,
              "Cauchy loss scale for loop-closure pairs in the video "
              "solver.");
  MakeDataclass(PyRotationEstimatorOptions);

  m.def(
      "run_rotation_averaging",
      [](const RotationEstimatorOptions& options,
         PoseGraph& pose_graph,
         Reconstruction& reconstruction,
         const std::vector<PosePrior>& pose_priors,
         const CorrespondenceGraph* correspondence_graph,
         bool extract_final_weights) {
        std::unordered_map<image_pair_t, double> final_weights;
        bool success = false;
        {
          py::gil_scoped_release release;
          success = RunRotationAveraging(
              options,
              pose_graph,
              reconstruction,
              pose_priors,
              extract_final_weights ? &final_weights : nullptr,
              correspondence_graph);
        }
        if (!extract_final_weights) {
          return py::cast(success);
        }
        py::dict result;
        result["success"] = success;
        py::dict weights_map;
        for (const auto& [pair_id, weight] : final_weights) {
          weights_map[py::cast(pair_id)] = weight;
        }
        result["final_weights"] = weights_map;
        return py::cast<py::object>(result);
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "pose_priors"_a,
      "correspondence_graph"_a = nullptr,
      "extract_final_weights"_a = false,
      "High-level rotation averaging solver that handles rig expansion. "
      "Returns True if rotation averaging succeeded. When "
      "``extract_final_weights=True``, returns ``{success, final_weights}`` "
      "dict instead. ``correspondence_graph`` is required when "
      "``options.skip_risky_LC_pairs=True`` so the LC-majority filter can "
      "read ImagePair.{inliers, are_lc} (PoseGraph::Edge does not carry "
      "them).");
}

void BindMotionAveraging(py::module& m) {
  BindGravityRefiner(m);
  BindRotationEstimator(m);
  BindGlobalPositioner(m);
}
