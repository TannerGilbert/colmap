COLMAP
======

About
-----

COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline with a graphical and command-line interface. It offers a wide
range of features for reconstruction of ordered and unordered image collections.
The software is licensed under the new BSD license.

The latest source code is available at https://github.com/colmap/colmap. COLMAP
builds on top of existing works and when using specific algorithms within
COLMAP, please also cite the original authors, as specified in the source code,
and consider citing relevant third-party dependencies (most notably
ceres-solver, poselib, sift-gpu, vlfeat).

Native Sequential LC Branch
---------------------------

This branch is for a student/internal native COLMAP loop-closure workflow. It is
not proposed upstream COLMAP behavior. The branch keeps the normal sequential
matching workflow, but can mark selected verified pairs as loop closures in the
database so downstream experiments can distinguish local tracking pairs from LC
pairs.

Build from the repository root:

    bash scripts/build_cpp.sh

Minimal CLI workflow:

    COLMAP="$PWD/local/bin/colmap"
    DB=/path/to/database.db
    IMAGES=/path/to/images
    SPARSE=/path/to/sparse
    VOCAB_TREE=/path/to/vocab_tree.bin

    "$COLMAP" feature_extractor \
        --database_path "$DB" \
        --image_path "$IMAGES"

    "$COLMAP" sequential_matcher \
        --database_path "$DB" \
        --SequentialMatching.overlap 10 \
        --SequentialMatching.loop_detection 1 \
        --SequentialMatching.vocab_tree_path "$VOCAB_TREE" \
        --SequentialMatching.loop_detection_period 10 \
        --SequentialMatching.loop_detection_num_images 50 \
        --SequentialMatching.mark_loop_detection_as_lc 1 \
        --SequentialMatching.mark_non_consecutive_as_lc 1

    mkdir -p "$SPARSE"
    "$COLMAP" global_mapper \
        --database_path "$DB" \
        --image_path "$IMAGES" \
        --output_path "$SPARSE" \
        --GlobalMapper.track_lc_second_pass 1 \
        --GlobalMapper.gp_use_lc_observations 1

The LC-specific flags are:

* ``SequentialMatching.loop_detection`` enables vocabulary-tree loop detection
  inside the sequential matcher.
* ``SequentialMatching.vocab_tree_path`` points to the vocabulary tree used for
  retrieval.
* ``SequentialMatching.loop_detection_period`` runs loop detection every N
  images.
* ``SequentialMatching.loop_detection_num_images`` is the number of retrieved
  candidate images before optional spatial-verification pruning.
* ``SequentialMatching.mark_loop_detection_as_lc`` stores verified
  loop-detection pairs as LC provenance in ``two_view_geometries``.
* ``SequentialMatching.mark_non_consecutive_as_lc`` also marks non-consecutive
  sequential-overlap pairs as LC candidates; consecutive neighbors remain normal
  local pairs.
* ``GlobalMapper.track_lc_second_pass`` appends LC observations to established
  tracks after the regular tracking pass.
* ``GlobalMapper.gp_use_lc_observations`` lets global positioning consume those
  LC observations.

Inspect LC provenance with SQLite:

    sqlite3 "$DB" \
        "SELECT is_loop_closure, COUNT(*) FROM two_view_geometries GROUP BY is_loop_closure;"

    sqlite3 "$DB" \
        "SELECT COUNT(*) FROM two_view_geometries WHERE is_loop_closure = 1;"

To list a few LC image pairs by name:

    sqlite3 "$DB" "
    SELECT i1.name, i2.name
    FROM two_view_geometries AS tvg
    JOIN images AS i1
      ON i1.image_id = CAST(tvg.pair_id / 2147483647 AS INTEGER)
    JOIN images AS i2
      ON i2.image_id = tvg.pair_id % 2147483647
    WHERE tvg.is_loop_closure = 1
    LIMIT 20;"

For mixed-provenance pairs, ``inlier_matches_are_lc`` contains one byte per
inlier match. Most quick checks only need ``is_loop_closure``.

Download
--------

* Binaries for **Windows** and other resources can be downloaded
  from https://github.com/colmap/colmap/releases.
* Binaries for **Linux/Unix/BSD** are available at
  https://repology.org/metapackage/colmap/versions.
* Pre-built **Docker** images are available at
  https://hub.docker.com/r/colmap/colmap.
* Conda packages are available at https://anaconda.org/conda-forge/colmap and
  can be installed with `conda install colmap`
* **Python bindings** are available at https://pypi.org/project/pycolmap.
  CUDA-enabled wheels are available at https://pypi.org/project/pycolmap-cuda12.
* To **build from source**, please see https://colmap.github.io/install.html.

Getting Started
---------------

1. Download pre-built binaries or build from source.
2. Download one of the provided [sample datasets](https://demuc.de/colmap/datasets/)
   or use your own images.
3. Use the **automatic reconstruction** to easily build models
   with a single click or command.

Documentation
-------------

The documentation is available [here](https://colmap.github.io/).

To build and update the documentation at the documentation website,
follow [these steps](https://colmap.github.io/install.html#documentation).

Support
-------

Please, use [GitHub Discussions](https://github.com/colmap/colmap/discussions)
for questions and the [GitHub issue tracker](https://github.com/colmap/colmap)
for bug reports, feature requests/additions, etc.

Acknowledgments
---------------

COLMAP was originally written by [Johannes Schönberger](https://demuc.de/) with
funding provided by his PhD advisors Jan-Michael Frahm and Marc Pollefeys.
The team of core project maintainers currently includes
[Johannes Schönberger](https://github.com/ahojnnes),
[Paul-Edouard Sarlin](https://github.com/sarlinpe),
[Shaohui Liu](https://github.com/B1ueber2y), and
[Linfei Pan](https://lpanaf.github.io/).

The Python bindings in PyCOLMAP were originally added by
[Mihai Dusmanu](https://github.com/mihaidusmanu),
[Philipp Lindenberger](https://github.com/Phil26AT), and
[Paul-Edouard Sarlin](https://github.com/sarlinpe).

The project has also benefitted from countless community contributions, including
bug fixes, improvements, new features, third-party tooling, and community
support (special credits to [Torsten Sattler](https://tsattler.github.io)).

Citation
--------

If you use this project for your research, please cite:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }

If you use the global SfM pipeline (GLOMAP), please cite:

    @inproceedings{pan2024glomap,
        author={Pan, Linfei and Barath, Daniel and Pollefeys, Marc and Sch\"{o}nberger, Johannes Lutz},
        title={{Global Structure-from-Motion Revisited}},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2024},
    }

If you use the image retrieval / vocabulary tree engine, please cite:

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }

Contribution
------------

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and
should be submitted in the form of new issues and/or pull requests on GitHub.

License
-------

The COLMAP library is licensed under the new BSD license. Note that this text
refers only to the license for COLMAP itself, independent of its thirdparty
dependencies, which are separately licensed. Building COLMAP with these
dependencies may affect the resulting COLMAP license.

    Copyright (c), ETH Zurich and UNC Chapel Hill.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
          its contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
