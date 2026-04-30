#include "colmap/estimators/view_graph_calibration.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindViewGraphCalibration(py::module& m) {
  auto PyFocalLengthCalibInput =
      py::classh<FocalLengthCalibInput>(m, "FocalLengthCalibInput")
          .def(py::init<>())
          .def_readwrite("pair_id", &FocalLengthCalibInput::pair_id)
          .def_readwrite("camera_id1", &FocalLengthCalibInput::camera_id1)
          .def_readwrite("camera_id2", &FocalLengthCalibInput::camera_id2)
          .def_readwrite("F", &FocalLengthCalibInput::F);
  MakeDataclass(PyFocalLengthCalibInput);

  auto PyFocalLengthCalibResult =
      py::classh<FocalLengthCalibResult>(m, "FocalLengthCalibResult")
          .def(py::init<>())
          .def_readwrite("focal_lengths",
                         &FocalLengthCalibResult::focal_lengths)
          .def_readwrite("calibration_errors_sq",
                         &FocalLengthCalibResult::calibration_errors_sq)
          .def_readwrite("success", &FocalLengthCalibResult::success);
  MakeDataclass(PyFocalLengthCalibResult);

  m.def(
      "calibrate_focal_lengths",
      [](const ViewGraphCalibrationOptions& options,
         const std::vector<FocalLengthCalibInput>& inputs,
         const std::unordered_map<camera_t, Camera>& cameras) {
        py::gil_scoped_release release;
        return CalibrateFocalLengths(options, inputs, cameras);
      },
      "options"_a,
      "inputs"_a,
      "cameras"_a,
      "Run Ceres focal-length optimization from fundamental matrices. "
      "Returns FocalLengthCalibResult with optimized focal lengths and "
      "per-pair calibration errors. Pure function — does not mutate any "
      "scene state.");
}
