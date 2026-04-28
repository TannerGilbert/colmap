#include "colmap/image/undistortion.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindUndistortion(py::module& m) {
  auto PyUndistortCameraOptions =
      py::classh<UndistortCameraOptions>(m, "UndistortCameraOptions")
          .def(py::init<>())
          .def_readwrite("blank_pixels", &UndistortCameraOptions::blank_pixels)
          .def_readwrite("min_scale", &UndistortCameraOptions::min_scale)
          .def_readwrite("max_scale", &UndistortCameraOptions::max_scale)
          .def_readwrite("max_image_size",
                         &UndistortCameraOptions::max_image_size)
          .def_readwrite("roi_min_x", &UndistortCameraOptions::roi_min_x)
          .def_readwrite("roi_min_y", &UndistortCameraOptions::roi_min_y)
          .def_readwrite("roi_max_x", &UndistortCameraOptions::roi_max_x)
          .def_readwrite("roi_max_y", &UndistortCameraOptions::roi_max_y);
  MakeDataclass(PyUndistortCameraOptions);

  m.def("undistort_camera",
        &UndistortCamera,
        "options"_a,
        "camera"_a,
        "Undistort camera.");

  m.def(
      "undistort_image",
      [](const UndistortCameraOptions& options,
         const Bitmap& distorted_image,
         const Camera& distorted_camera) -> std::pair<Bitmap, Camera> {
        py::gil_scoped_release release;
        std::pair<Bitmap, Camera> result;
        UndistortImage(options,
                       distorted_image,
                       distorted_camera,
                       &result.first,
                       &result.second);
        return result;
      },
      "options"_a,
      "distorted_image"_a,
      "distorted_camera"_a,
      "Undistort image and corresponding camera.");

  m.def(
      "undistort_images",
      [](py::dict cameras_py,
         py::dict images_py,
         bool clean_points) {
        // Convert dicts to C++ maps. Holding the GIL so we can iterate Python.
        std::unordered_map<camera_t, Camera> cameras;
        cameras.reserve(cameras_py.size());
        for (auto item : cameras_py) {
          cameras.emplace(py::cast<camera_t>(item.first),
                          py::cast<Camera>(item.second));
        }
        std::unordered_map<image_t, Image> images;
        images.reserve(images_py.size());
        for (auto item : images_py) {
          images.emplace(py::cast<image_t>(item.first),
                         py::cast<Image>(item.second));
        }

        {
          py::gil_scoped_release release;
          UndistortImageFeatures(cameras, images, clean_points);
        }

        // Build a fresh Python dict (one Image per entry, deep copy via
        // py::cast) so mutations to the C++ map propagate back. Avoid
        // returning the C++ map directly — pybind11's STL caster + classh
        // shared_ptr holder can move-from the values, leaving features
        // empty in the resulting Python instances.
        py::dict out;
        for (auto& [iid, img] : images) {
          out[py::cast(iid)] = py::cast(img);
        }
        return out;
      },
      "cameras"_a,
      "images"_a,
      "clean_points"_a = true,
      "Populate Image.features_undist (normalized 3D bearing rays) for every "
      "feature in Image.features. Returns a new images dict; rebind the "
      "caller's reference to receive the updated state.");
}
