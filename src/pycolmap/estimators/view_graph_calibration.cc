#include "colmap/estimators/view_graph_calibration.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindViewGraphCalibration(py::module& m) {
  m.def(
      "run_view_graph_calibration",
      [](CorrespondenceGraph& view_graph,
         Reconstruction& rec,
         const ViewGraphCalibrationOptions& options) {
        py::gil_scoped_release release;
        const bool success = CalibrateViewGraph(options, view_graph, rec);
        THROW_CHECK(success) << "Failed to solve view graph calibration.";
      },
      "view_graph"_a,
      "rec"_a,
      "options"_a,
      "Run view graph focal-length calibration on a CorrespondenceGraph + "
      "Reconstruction. Mutates rec (camera params) and view_graph (pair "
      "validity) in place. Bypasses CalibrateViewGraph's higher-level "
      "wrapper (cross-validation, F/E recomputation, etc.).");
}
