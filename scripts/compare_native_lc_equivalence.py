#!/usr/bin/env python3
"""Compare native sequential LC provenance against an independent DB reference.

The expected workflow is:
1. Create a baseline DB with sequential matching run with
   ``mark_non_consecutive_as_lc=0``.
2. Copy that DB, remove non-consecutive TVGs, and rerun sequential matching
   with ``mark_non_consecutive_as_lc=1``.
3. Run this script on both DBs.

The reference uses only adjacent baseline TVGs to reconstruct transitive
non-LC rows, then applies the same endpoint rule as VideoSfM's ``merge_matches``:
append candidate rows only when both endpoints are new relative to the
transitive rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap


def _rows(two_view_geometry):
    return [tuple(map(int, row)) for row in two_view_geometry.inlier_matches.tolist()]


def _make_two_view_geometry(template, rows, mask):
    tvg = pycolmap.TwoViewGeometry()
    tvg.config = template.config
    tvg.E = template.E
    tvg.F = template.F
    tvg.H = template.H
    tvg.cam2_from_cam1 = template.cam2_from_cam1
    tvg.tri_angle = template.tri_angle
    tvg.inlier_matches = np.asarray(rows, dtype=np.uint32).reshape((-1, 2))
    tvg.inlier_matches_are_lc = list(mask)
    tvg.is_loop_closure = any(mask)
    return tvg


def _optional_matrix_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return np.allclose(np.asarray(left), np.asarray(right), rtol=0, atol=1e-12)


def _optional_rigid_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return np.allclose(
        np.asarray(left.params), np.asarray(right.params), rtol=0, atol=1e-12
    )


def _two_view_geometry_metadata_mismatch(expected, native):
    mismatches = []
    if native.config != expected.config:
        mismatches.append(
            {
                "field": "config",
                "expected": int(expected.config),
                "native": int(native.config),
            }
        )
    for field in ("E", "F", "H"):
        if not _optional_matrix_equal(getattr(expected, field), getattr(native, field)):
            mismatches.append({"field": field})
    if not _optional_rigid_equal(expected.cam2_from_cam1, native.cam2_from_cam1):
        mismatches.append({"field": "cam2_from_cam1"})
    if not np.isclose(native.tri_angle, expected.tri_angle, rtol=0, atol=1e-12):
        mismatches.append(
            {
                "field": "tri_angle",
                "expected": float(expected.tri_angle),
                "native": float(native.tri_angle),
            }
        )
    return mismatches


def _build_adjacent_index(db, images):
    correspondences = {
        image.image_id: [[] for _ in range(db.num_keypoints_for_image(image.image_id))]
        for image in images
    }
    for idx in range(len(images) - 1):
        image1 = images[idx]
        image2 = images[idx + 1]
        tvg = db.read_two_view_geometry(image1.image_id, image2.image_id)
        for point2d_idx1, point2d_idx2 in _rows(tvg):
            if point2d_idx1 >= len(
                correspondences[image1.image_id]
            ) or point2d_idx2 >= len(correspondences[image2.image_id]):
                continue
            corr1 = correspondences[image1.image_id][point2d_idx1]
            if (image2.image_id, point2d_idx2) in corr1:
                continue
            corr1.append((image2.image_id, point2d_idx2))
            correspondences[image2.image_id][point2d_idx2].append(
                (image1.image_id, point2d_idx1)
            )
    return correspondences


def _extract_transitive_correspondences(correspondences, image_id, point2d_idx):
    if not correspondences[image_id][point2d_idx]:
        return []

    queue = [(image_id, point2d_idx)]
    visited = {image_id: {point2d_idx}}
    queue_beg = 0
    queue_end = 1

    while queue_beg < queue_end:
        for i in range(queue_beg, queue_end):
            ref_image_id, ref_point2d_idx = queue[i]
            for corr_image_id, corr_point2d_idx in correspondences[ref_image_id][
                ref_point2d_idx
            ]:
                image_visited = visited.setdefault(corr_image_id, set())
                if corr_point2d_idx not in image_visited:
                    image_visited.add(corr_point2d_idx)
                    queue.append((corr_image_id, corr_point2d_idx))
        queue_beg = queue_end
        queue_end = len(queue)

    if len(queue) > 1:
        queue[0] = queue[-1]
    queue.pop()
    return queue


def _transitive_matches(correspondences, db, image1, image2):
    matches = []
    used1 = set()
    used2 = set()
    for point2d_idx1 in range(db.num_keypoints_for_image(image1.image_id)):
        transitive_correspondences = _extract_transitive_correspondences(
            correspondences, image1.image_id, point2d_idx1
        )
        for corr_image_id, point2d_idx2 in transitive_correspondences:
            if corr_image_id != image2.image_id:
                continue
            if point2d_idx1 not in used1 and point2d_idx2 not in used2:
                matches.append((point2d_idx1, point2d_idx2))
                used1.add(point2d_idx1)
                used2.add(point2d_idx2)
            break
    return matches


def _expected_merge(transitive_rows, candidate_rows):
    transitive_points1 = {row[0] for row in transitive_rows}
    transitive_points2 = {row[1] for row in transitive_rows}
    accepted_candidates = [
        row
        for row in candidate_rows
        if row[0] not in transitive_points1 and row[1] not in transitive_points2
    ]
    return transitive_rows + accepted_candidates, [False] * len(transitive_rows) + [
        True
    ] * len(accepted_candidates)


def _add_images_to_graph(graph, db, images):
    for image in images:
        graph.add_image(image.image_id, db.num_keypoints_for_image(image.image_id))


def _build_graph(db, images, pair_geometries):
    graph = pycolmap.CorrespondenceGraph()
    _add_images_to_graph(graph, db, images)
    for image1, image2, tvg in pair_geometries:
        if tvg.inlier_matches.shape[0] == 0:
            continue
        graph.add_two_view_geometry(image1.image_id, image2.image_id, tvg)
    graph.finalize()
    return graph


def _image_pair_rows(image_pair):
    return [tuple(map(int, row)) for row in image_pair.matches.tolist()]


def _graph_pair_snapshot(graph, pair_id):
    image_pair = graph.image_pairs[pair_id]
    return {
        "matches": _image_pair_rows(image_pair),
        "inliers": list(map(int, image_pair.inliers)),
        "are_lc": list(map(bool, image_pair.are_lc)),
    }


def _compare_graphs(reference_graph, native_graph):
    reference_ids = sorted(reference_graph.image_pair_ids())
    native_ids = sorted(native_graph.image_pair_ids())
    failures = []
    if reference_ids != native_ids:
        failures.append(
            {
                "graph_pair_ids_mismatch": {
                    "reference_only": sorted(set(reference_ids) - set(native_ids)),
                    "native_only": sorted(set(native_ids) - set(reference_ids)),
                }
            }
        )

    for pair_id in sorted(set(reference_ids) & set(native_ids)):
        reference_snapshot = _graph_pair_snapshot(reference_graph, pair_id)
        native_snapshot = _graph_pair_snapshot(native_graph, pair_id)
        if reference_snapshot != native_snapshot:
            image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            failures.append(
                {
                    "pair": (int(image_id1), int(image_id2)),
                    "reference_head": {
                        "matches": reference_snapshot["matches"][:20],
                        "inliers": reference_snapshot["inliers"][:40],
                        "are_lc": reference_snapshot["are_lc"][:40],
                    },
                    "native_head": {
                        "matches": native_snapshot["matches"][:20],
                        "inliers": native_snapshot["inliers"][:40],
                        "are_lc": native_snapshot["are_lc"][:40],
                    },
                }
            )
    reference_image_ids = {
        image_id
        for pair_id in reference_ids
        for image_id in pycolmap.pair_id_to_image_pair(pair_id)
    }
    native_image_ids = {
        image_id
        for pair_id in native_ids
        for image_id in pycolmap.pair_id_to_image_pair(pair_id)
    }
    for image_id in sorted(reference_image_ids | native_image_ids):
        reference_counts = (
            reference_graph.num_observations_for_image(image_id),
            reference_graph.num_correspondences_for_image(image_id),
        )
        native_counts = (
            native_graph.num_observations_for_image(image_id),
            native_graph.num_correspondences_for_image(image_id),
        )
        if reference_counts != native_counts:
            failures.append(
                {
                    "image_graph_counts_mismatch": {
                        "image_id": int(image_id),
                        "reference": tuple(map(int, reference_counts)),
                        "native": tuple(map(int, native_counts)),
                    }
                }
            )
    return failures


def _total_matches(graph):
    matches_by_pair = graph.num_matches_between_all_images()
    if isinstance(matches_by_pair, dict):
        return sum(matches_by_pair.values())
    return matches_by_pair


def _read_tvg_pair_ids(db):
    pair_ids, _ = db.read_two_view_geometries()
    return set(pair_ids)


def _assert_baseline_has_no_lc_provenance(db):
    pair_ids, two_view_geometries = db.read_two_view_geometries()
    for pair_id, tvg in zip(pair_ids, two_view_geometries):
        mask = list(map(bool, tvg.inlier_matches_are_lc))
        if tvg.is_loop_closure or any(mask):
            raise RuntimeError(
                f"baseline DB is polluted with LC provenance for pair_id={pair_id}"
            )


def _camera_signature(camera):
    return (
        camera.camera_id,
        str(camera.model),
        camera.width,
        camera.height,
        tuple(np.asarray(camera.params).tolist()),
        camera.has_prior_focal_length,
    )


def _assert_same_cameras(baseline_db, native_db):
    baseline_cameras = sorted(
        (_camera_signature(camera) for camera in baseline_db.read_all_cameras()),
        key=lambda item: item[0],
    )
    native_cameras = sorted(
        (_camera_signature(camera) for camera in native_db.read_all_cameras()),
        key=lambda item: item[0],
    )
    if native_cameras != baseline_cameras:
        raise RuntimeError("baseline and native DBs must have identical cameras")


def _assert_same_raw_matches(baseline_db, native_db):
    baseline_pair_ids, baseline_matches = baseline_db.read_all_matches()
    native_pair_ids, native_matches = native_db.read_all_matches()
    if baseline_pair_ids != native_pair_ids:
        raise RuntimeError("baseline and native DBs must have identical match pairs")
    for pair_id, baseline_rows, native_rows in zip(
        baseline_pair_ids, baseline_matches, native_matches
    ):
        if not np.array_equal(native_rows, baseline_rows):
            raise RuntimeError(
                f"baseline and native DBs differ in raw matches for pair_id={pair_id}"
            )


def _image_positions(images):
    return {image.image_id: idx for idx, image in enumerate(images)}


def _image_by_id(images):
    return {image.image_id: image for image in images}


def _pair_distance(pair_id, positions):
    image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
    if image_id1 not in positions or image_id2 not in positions:
        return None
    return abs(positions[image_id2] - positions[image_id1])


def _sequence_ordered_pair(pair_id, positions, images_by_id):
    image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
    if image_id1 not in positions or image_id2 not in positions:
        return None
    if positions[image_id1] <= positions[image_id2]:
        return images_by_id[image_id1], images_by_id[image_id2]
    return images_by_id[image_id2], images_by_id[image_id1]


def _assert_same_images_and_keypoints(baseline_db, native_db, images, native_images):
    baseline_signature = [
        (
            image.image_id,
            image.name,
            image.camera_id,
            image.frame_id,
            baseline_db.num_keypoints_for_image(image.image_id),
        )
        for image in images
    ]
    native_signature = [
        (
            image.image_id,
            image.name,
            image.camera_id,
            image.frame_id,
            native_db.num_keypoints_for_image(image.image_id),
        )
        for image in native_images
    ]
    if native_signature != baseline_signature:
        raise RuntimeError(
            "baseline and native DBs must have identical image ids, names, "
            "and keypoint counts"
        )
    for image in images:
        baseline_keypoints = baseline_db.read_keypoints(image.image_id)
        native_keypoints = native_db.read_keypoints(image.image_id)
        if not np.array_equal(native_keypoints, baseline_keypoints):
            raise RuntimeError(
                "baseline and native DBs must have identical keypoint rows"
            )

        baseline_descriptors = baseline_db.read_descriptors(image.image_id)
        native_descriptors = native_db.read_descriptors(image.image_id)
        if native_descriptors.type != baseline_descriptors.type or not np.array_equal(
            native_descriptors.data, baseline_descriptors.data
        ):
            raise RuntimeError(
                "baseline and native DBs must have identical descriptor rows"
            )


def compare(baseline_db_path: Path, native_db_path: Path, max_distance: int) -> int:
    print({"pycolmap": pycolmap.__file__})
    if baseline_db_path.resolve() == native_db_path.resolve():
        raise RuntimeError("baseline and native DB paths must be different")

    baseline_db = pycolmap.Database.open(baseline_db_path)
    native_db = pycolmap.Database.open(native_db_path)
    try:
        images = sorted(baseline_db.read_all_images(), key=lambda image: image.name)
        native_images = sorted(
            native_db.read_all_images(), key=lambda image: image.name
        )
        _assert_same_images_and_keypoints(baseline_db, native_db, images, native_images)
        _assert_same_cameras(baseline_db, native_db)
        _assert_same_raw_matches(baseline_db, native_db)
        _assert_baseline_has_no_lc_provenance(baseline_db)

        positions = _image_positions(images)
        images_by_id = _image_by_id(images)
        baseline_db_pair_ids = _read_tvg_pair_ids(baseline_db)
        native_db_pair_ids = _read_tvg_pair_ids(native_db)
        adjacent_index = _build_adjacent_index(baseline_db, images)
        failures = []
        checked = []
        non_adjacent_checked = 0
        accepted_lc_rows = 0
        expected_geometries = []
        native_geometries = []

        for idx in range(len(images) - 1):
            image1 = images[idx]
            image2 = images[idx + 1]
            baseline_tvg = baseline_db.read_two_view_geometry(
                image1.image_id, image2.image_id
            )
            native_tvg = native_db.read_two_view_geometry(
                image1.image_id, image2.image_id
            )
            baseline_rows = _rows(baseline_tvg)
            native_rows = _rows(native_tvg)
            native_mask = list(map(bool, native_tvg.inlier_matches_are_lc))
            expected_tvg = _make_two_view_geometry(
                baseline_tvg, baseline_rows, [False] * len(baseline_rows)
            )
            metadata_mismatches = _two_view_geometry_metadata_mismatch(
                expected_tvg, native_tvg
            )
            ok = (
                native_rows == baseline_rows
                and not any(native_mask)
                and not native_tvg.is_loop_closure
                and not metadata_mismatches
            )
            checked.append(
                {
                    "pair": (image1.name, image2.name),
                    "distance": 1,
                    "expected_false": len(baseline_rows),
                    "expected_true": 0,
                    "native_false": len(native_rows) - native_mask.count(True),
                    "native_true": native_mask.count(True),
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(
                    {
                        "pair": (image1.name, image2.name),
                        "distance": 1,
                        "expected_rows_head": baseline_rows[:20],
                        "native_rows_head": native_rows[:20],
                        "native_mask_head": native_mask[:40],
                        "metadata_mismatches": metadata_mismatches,
                    }
                )
            if baseline_rows:
                expected_geometries.append((image1, image2, expected_tvg))
            if native_rows:
                native_geometries.append((image1, image2, native_tvg))

        non_adjacent_pair_ids = sorted(
            pair_id
            for pair_id in baseline_db_pair_ids | native_db_pair_ids
            if _pair_distance(pair_id, positions) != 1
        )
        for pair_id in non_adjacent_pair_ids:
            ordered_pair = _sequence_ordered_pair(pair_id, positions, images_by_id)
            if ordered_pair is None:
                failures.append({"unknown_image_pair": int(pair_id)})
                continue
            image1, image2 = ordered_pair
            distance = _pair_distance(pair_id, positions)
            if distance is None or distance > max_distance:
                failures.append(
                    {
                        "pair_outside_max_distance": {
                            "pair": (image1.name, image2.name),
                            "distance": distance,
                            "max_distance": max_distance,
                        }
                    }
                )
                continue

            baseline_tvg = baseline_db.read_two_view_geometry(
                image1.image_id, image2.image_id
            )
            candidate_rows = _rows(baseline_tvg)

            transitive_rows = _transitive_matches(
                adjacent_index, baseline_db, image1, image2
            )
            expected_rows, expected_mask = _expected_merge(
                transitive_rows, candidate_rows
            )
            expected_tvg = _make_two_view_geometry(
                baseline_tvg, expected_rows, expected_mask
            )

            native_tvg = native_db.read_two_view_geometry(
                image1.image_id, image2.image_id
            )
            native_rows = _rows(native_tvg)
            native_mask = list(map(bool, native_tvg.inlier_matches_are_lc))
            metadata_mismatches = _two_view_geometry_metadata_mismatch(
                expected_tvg, native_tvg
            )
            ok = (
                native_rows == expected_rows
                and native_mask == expected_mask
                and native_tvg.is_loop_closure == any(expected_mask)
                and not metadata_mismatches
            )
            if not expected_rows and not native_rows:
                continue
            non_adjacent_checked += 1
            accepted_lc_rows += expected_mask.count(True)
            checked.append(
                {
                    "pair": (image1.name, image2.name),
                    "distance": distance,
                    "candidate": len(candidate_rows),
                    "expected_false": len(transitive_rows),
                    "expected_true": len(expected_rows) - len(transitive_rows),
                    "native_false": native_mask.count(False),
                    "native_true": native_mask.count(True),
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(
                    {
                        "pair": (image1.name, image2.name),
                        "expected_rows_head": expected_rows[:20],
                        "native_rows_head": native_rows[:20],
                        "expected_mask_head": expected_mask[:40],
                        "native_mask_head": native_mask[:40],
                        "metadata_mismatches": metadata_mismatches,
                    }
                )
            if expected_rows:
                expected_geometries.append((image1, image2, expected_tvg))
            if native_rows:
                native_geometries.append((image1, image2, native_tvg))

        reference_graph = _build_graph(baseline_db, images, expected_geometries)
        native_graph = _build_graph(native_db, native_images, native_geometries)
        graph_failures = _compare_graphs(reference_graph, native_graph)
        failures.extend({"graph_failure": failure} for failure in graph_failures)

        expected_pair_ids = set(reference_graph.image_pair_ids())
        if expected_pair_ids != native_db_pair_ids:
            failures.append(
                {
                    "native_db_pair_ids_mismatch": {
                        "expected_only": sorted(expected_pair_ids - native_db_pair_ids),
                        "native_only": sorted(native_db_pair_ids - expected_pair_ids),
                    }
                }
            )
        if non_adjacent_checked == 0:
            failures.append({"vacuous_check": "no non-adjacent pair was checked"})
        if accepted_lc_rows == 0:
            failures.append({"vacuous_check": "no accepted LC inlier was checked"})

        print(
            {
                "checked_pairs": len(checked),
                "non_adjacent_checked": non_adjacent_checked,
                "accepted_lc_rows": accepted_lc_rows,
                "reference_graph_pairs": reference_graph.num_image_pairs(),
                "native_graph_pairs": native_graph.num_image_pairs(),
                "reference_graph_matches": _total_matches(reference_graph),
                "native_graph_matches": _total_matches(native_graph),
                "failures": len(failures),
            }
        )
        for row in checked:
            print(row)
        if failures:
            print({"failure_details": failures})
            return 1
        return 0
    finally:
        baseline_db.close()
        native_db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-db", required=True, type=Path)
    parser.add_argument("--native-db", required=True, type=Path)
    parser.add_argument("--max-distance", required=True, type=int)
    args = parser.parse_args()
    return compare(args.baseline_db, args.native_db, args.max_distance)


if __name__ == "__main__":
    raise SystemExit(main())
