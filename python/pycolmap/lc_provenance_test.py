from pathlib import Path

import numpy as np
import pycolmap


def test_lc_provenance_defaults() -> None:
    sequential_options = pycolmap.SequentialPairingOptions()
    assert not sequential_options.mark_loop_detection_as_lc
    assert not sequential_options.mark_non_consecutive_as_lc

    vocab_tree_options = pycolmap.VocabTreePairingOptions()
    assert not vocab_tree_options.mark_matches_as_lc

    global_positioner_options = pycolmap.GlobalPositionerOptions()
    assert not global_positioner_options.use_lc_observations

    rotation_options = pycolmap.RotationEstimatorOptions()
    assert not rotation_options.skip_risky_lc_pairs
    assert not rotation_options.use_video_constraints


def test_lc_provenance_bindings_roundtrip(tmp_path: Path) -> None:
    sequential_options = pycolmap.SequentialPairingOptions()
    sequential_options.mark_loop_detection_as_lc = True
    sequential_options.mark_non_consecutive_as_lc = True
    assert sequential_options.mark_loop_detection_as_lc
    assert sequential_options.mark_non_consecutive_as_lc

    vocab_tree_options = pycolmap.VocabTreePairingOptions()
    vocab_tree_options.mark_matches_as_lc = True
    assert vocab_tree_options.mark_matches_as_lc

    two_view_geometry = pycolmap.TwoViewGeometry()
    assert not two_view_geometry.is_loop_closure
    two_view_geometry.is_loop_closure = True
    assert two_view_geometry.is_loop_closure
    two_view_geometry.inlier_matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    two_view_geometry.inlier_matches_are_lc = [True, False]
    assert list(two_view_geometry.inlier_matches_are_lc) == [True, False]

    database = pycolmap.Database.open(tmp_path / "database.db")
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 640, 480
    )
    camera_id = database.write_camera(camera)
    image_id1 = database.write_image(
        pycolmap.Image(name="image1.jpg", camera_id=camera_id)
    )
    image_id2 = database.write_image(
        pycolmap.Image(name="image2.jpg", camera_id=camera_id)
    )

    database.write_two_view_geometry(image_id1, image_id2, two_view_geometry)
    read_two_view_geometry = database.read_two_view_geometry(image_id1, image_id2)
    assert read_two_view_geometry.is_loop_closure
    assert list(read_two_view_geometry.inlier_matches_are_lc) == [True, False]
    database.close()


def test_correspondence_graph_image_pair_are_lc_binding() -> None:
    image_pair = pycolmap.ImagePair(1, 2)
    image_pair.matches = np.array([[0, 0], [1, 1]], dtype=np.int32)
    image_pair.are_lc = [True, False]
    assert list(image_pair.are_lc) == [True, False]

    with np.testing.assert_raises(ValueError):
        image_pair.are_lc = [True]
