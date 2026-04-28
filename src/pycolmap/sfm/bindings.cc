#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindObservationManager(py::module& m);
void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);
void BindViewGraphManipulation(py::module& m);
void BindTrackEstablishment(py::module& m);
void BindImagePairInliers(py::module& m);
void BindTrackFilter(py::module& m);

void BindSfm(py::module& m) {
  BindObservationManager(m);
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
  BindViewGraphManipulation(m);
  BindTrackEstablishment(m);
  BindImagePairInliers(m);
  BindTrackFilter(m);
}
