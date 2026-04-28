#include "colmap/scene/image.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/point2d.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>
#include <optional>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename T>
std::shared_ptr<Image> MakeImage(const std::string& name,
                                 const std::vector<T>& points2D,
                                 camera_t camera_id,
                                 image_t image_id) {
  auto image = std::make_shared<Image>();
  image->SetName(name);
  image->SetPoints2D(points2D);
  if (camera_id != kInvalidCameraId) {
    image->SetCameraId(camera_id);
  }
  image->SetImageId(image_id);
  return image;
}

void BindSceneImage(py::module& m) {
  py::classh<Image> PyImage(m, "Image");
  PyImage.def(py::init<>())
      .def(py::init(&MakeImage<Point2D>),
           "name"_a = "",
           py::arg_v("points2D", Point2DVector(), "Point2DList()"),
           py::arg_v(
               "camera_id", kInvalidCameraId, "pycolmap.INVALID_CAMERA_ID"),
           py::arg_v("image_id", kInvalidImageId, "pycolmap.INVALID_IMAGE_ID"))
      .def(py::init(&MakeImage<Eigen::Vector2d>),
           "name"_a = "",
           "keypoints"_a = std::vector<Eigen::Vector2d>(),
           py::arg_v(
               "camera_id", kInvalidCameraId, "pycolmap.INVALID_CAMERA_ID"),
           py::arg_v("image_id", kInvalidImageId, "pycolmap.INVALID_IMAGE_ID"))
      .def_property("image_id",
                    &Image::ImageId,
                    &Image::SetImageId,
                    "Unique identifier of the image.")
      .def_property("camera_id",
                    &Image::CameraId,
                    &Image::SetCameraId,
                    "Unique identifier of the camera.")
      .def_property("frame_id",
                    &Image::FrameId,
                    &Image::SetFrameId,
                    "Unique identifier of the frame.")
      .def_property_readonly(
          "data_id", &Image::DataId, "Unique identifier of the data.")
      .def_property(
          "camera",
          [](Image& self) -> py::typing::Optional<Camera> {
            if (self.HasCameraPtr()) {
              return py::cast(self.CameraPtr());
            } else {
              return py::none();
            }
          },
          &Image::SetCameraPtr,
          "The associated camera object.")
      .def_property(
          "frame",
          [](Image& self) -> py::typing::Optional<Frame> {
            if (self.HasFramePtr()) {
              return py::cast(self.FramePtr());
            } else {
              return py::none();
            }
          },
          &Image::SetFramePtr,
          "The associated frame object.")
      .def_property("name",
                    py::overload_cast<>(&Image::Name),
                    &Image::SetName,
                    "Name of the image.")
      // Pose accessor — reads Frame-derived pose; returns identity Rigid3d for
      // images without a posed Frame (not yet registered). Writes update the
      // Frame's rig_from_world; requires the image to have a Frame attached.
      // Use Reconstruction.frame(img.frame_id).rig_from_world for bulk writes.
      .def_property(
          "cam_from_world",
          [](const Image& self) -> Rigid3d {
            return self.HasPose() ? self.CamFromWorld() : Rigid3d();
          },
          [](Image& self, const Rigid3d& value) {
            THROW_CHECK(self.HasFramePtr())
                << "Cannot set cam_from_world on an Image without an attached "
                   "Frame. Add the image to a Reconstruction first, then write "
                   "via rec.frame(img.frame_id).rig_from_world.";
            self.FramePtr()->SetRigFromWorld(value);
          },
          "Pose of the image (cam_from_world). Reads the Frame-derived pose "
          "when available, else identity. Setter requires the image to be "
          "attached to a Frame (i.e. inside a Reconstruction).")
      .def_property_readonly(
          "has_pose", &Image::HasPose, "Whether the image has a valid pose.")
      .def_property(
          "points2D",
          py::overload_cast<>(&Image::Points2D),
          py::overload_cast<const Point2DVector&>(&Image::SetPoints2D),
          py::return_value_policy::reference_internal,
          "Array of Points2D (=keypoints).")
      .def("point2D",
           py::overload_cast<camera_t>(&Image::Point2D),
           py::return_value_policy::reference_internal,
           "point2D_idx"_a,
           "Direct accessor for a point2D.")
      .def("point2D_coords",
           &Image::Point2DCoords,
           "Get an Nx2 numpy array of xy coordinates for points2D.")
      .def("point3D_ids",
           &Image::Point3DIds,
           "Get a list of 3D point IDs for all points2D. "
           "Returns kInvalidPoint3DId for untriangulated points.")
      .def_property(
          "pixel_cholesky_xy",
          [](const Image& self) { return self.PixelCholeskyXY(); },
          [](Image& self, const std::vector<Eigen::Vector3d>& v) {
            self.SetPixelCholeskyXY(v);
          },
          "Per-observation Cholesky factors (L00, L10, L11) for pixel "
          "covariance weighting.")
      .def("has_pixel_covariances",
           &Image::HasPixelCovariances,
           "Check if pixel covariances are set and match points2D count.")
      // Per-feature/per-image fields. Bound as def_property with Eigen-typed
      // getters/setters so Python sees numpy.ndarray rather than list — the
      // pipeline code does numpy fancy-indexing on these.
      .def_property(
          "is_inlier",
          [](const Image& self) -> Eigen::Array<bool, Eigen::Dynamic, 1> {
            Eigen::Array<bool, Eigen::Dynamic, 1> arr(self.is_inlier.size());
            for (size_t i = 0; i < self.is_inlier.size(); ++i)
              arr[i] = self.is_inlier[i];
            return arr;
          },
          [](Image& self,
             const Eigen::Array<bool, Eigen::Dynamic, 1>& v) {
            self.is_inlier.assign(v.size(), false);
            for (Eigen::Index i = 0; i < v.size(); ++i)
              self.is_inlier[i] = v[i];
          })
      .def_property(
          "is_track_anchor",
          [](const Image& self) -> Eigen::Array<bool, Eigen::Dynamic, 1> {
            Eigen::Array<bool, Eigen::Dynamic, 1> arr(
                self.is_track_anchor.size());
            for (size_t i = 0; i < self.is_track_anchor.size(); ++i)
              arr[i] = self.is_track_anchor[i];
            return arr;
          },
          [](Image& self,
             const Eigen::Array<bool, Eigen::Dynamic, 1>& v) {
            self.is_track_anchor.assign(v.size(), false);
            for (Eigen::Index i = 0; i < v.size(); ++i)
              self.is_track_anchor[i] = v[i];
          })
      .def_readwrite("angular_stddevs", &Image::angular_stddevs)
      .def_readwrite("angular_cholesky_xy", &Image::angular_cholesky_xy)
      // FORK-REMOVAL TODO — `features` / `features_undist` are fork-only
      // fields. See `.claude/notes/glomap_audit/fork_removal_todo.md`.
      .def_readwrite("features", &Image::features)
      .def_readwrite("features_undist", &Image::features_undist)
      .def(
          "set_point3D_for_point2D",
          &Image::SetPoint3DForPoint2D,
          "point2D_Idx"_a,
          "point3D_id"_a,
          "Set the point as triangulated, i.e. it is part of a 3D point track.")
      .def("reset_point3D_for_point2D",
           &Image::ResetPoint3DForPoint2D,
           "point2D_idx"_a,
           "Set the point as not triangulated, i.e. it is not part of a 3D "
           "point track")
      .def("has_point3D",
           &Image::HasPoint3D,
           "point3D_id"_a,
           "Check whether one of the image points is part of a 3D point track.")
      .def("projection_center",
           &Image::ProjectionCenter,
           "Extract the projection center in world space.")
      .def("viewing_direction",
           &Image::ViewingDirection,
           "Extract the viewing direction of the image.")
      .def("project_point",
           &Image::ProjectPoint,
           "Project 3D point onto the image",
           "point3D"_a)
      .def("has_camera_id",
           &Image::HasCameraId,
           "Check whether identifier of camera has been set.")
      .def("has_camera_ptr",
           &Image::HasCameraPtr,
           "Check whether the camera pointer has been set.")
      .def("reset_camera_ptr",
           &Image::ResetCameraPtr,
           "Make the camera pointer a nullptr.")
      .def("has_frame_id",
           &Image::HasFrameId,
           "Check whether identifier of frame has been set.")
      .def("has_frame_ptr",
           &Image::HasFramePtr,
           "Check whether the frame pointer has been set.")
      .def("reset_frame_ptr",
           &Image::ResetFramePtr,
           "Make the frame pointer a nullptr.")
      .def(
          "is_ref_in_frame",
          &Image::IsRefInFrame,
          "Check if the image was captured by the reference sensor in the rig.")
      .def("num_points2D",
           &Image::NumPoints2D,
           "Get the number of image points (keypoints).")
      .def_property_readonly(
          "num_points3D",
          &Image::NumPoints3D,
          "Get the number of triangulations, i.e. the number of points that\n"
          "are part of a 3D point track.")
      .def(
          "get_observation_point2D_idxs",
          [](const Image& self) {
            std::vector<point2D_t> point2D_idxs;
            for (point2D_t point2D_idx = 0; point2D_idx < self.NumPoints2D();
                 ++point2D_idx) {
              if (self.Point2D(point2D_idx).HasPoint3D()) {
                point2D_idxs.push_back(point2D_idx);
              }
            }
            return point2D_idxs;
          },
          "Get the indices of 2D points that observe a 3D point.")
      .def(
          "get_observation_points2D",
          [](const Image& self) {
            Point2DVector points2D;
            for (const auto& point2D : self.Points2D()) {
              if (point2D.HasPoint3D()) {
                points2D.push_back(point2D);
              }
            }
            return points2D;
          },
          "Get the 2D points that observe a 3D point.");
  MakeDataclass(PyImage);

  py::bind_map<ImageMap>(m, "ImageMap");
}
