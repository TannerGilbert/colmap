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

#include "colmap/sfm/image_pair_inliers.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindImagePairInliers(py::module& m) {
  auto PyOpts =
      py::classh<InlierThresholdOptions>(m, "InlierThresholdOptions")
          .def(py::init<>())
          .def_readwrite("max_epipolar_error_E",
                         &InlierThresholdOptions::max_epipolar_error_E)
          .def_readwrite("max_epipolar_error_F",
                         &InlierThresholdOptions::max_epipolar_error_F)
          .def_readwrite("max_epipolar_error_H",
                         &InlierThresholdOptions::max_epipolar_error_H)
          .def_readwrite("min_angle_from_epipole",
                         &InlierThresholdOptions::min_angle_from_epipole);
  MakeDataclass(PyOpts);

  m.def(
      "image_pairs_inlier_count",
      [](CorrespondenceGraph& correspondence_graph,
         Reconstruction& rec,
         const InlierThresholdOptions& options,
         bool clean_inliers) {
        py::gil_scoped_release release;
        ImagePairsInlierCount(correspondence_graph, rec, options, clean_inliers);
      },
      "correspondence_graph"_a,
      "rec"_a,
      "options"_a,
      "clean_inliers"_a,
      "Per-pair inlier scoring (Sampson + cheirality + epipole checks). "
      "Updates ImagePair.inliers and is_valid in place.");
}
