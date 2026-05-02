#!/usr/bin/env python3
"""Smoke-test the native sequential LC CLI path without external data."""

from __future__ import annotations

import argparse
import random
import shlex
import sqlite3
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLMAP = REPO_ROOT / "local" / "bin" / "colmap"
IMAGE_PAIR_ID_MAX = 2_147_483_647


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_png_rgb(path: Path, width: int, height: int, rgb: bytes) -> None:
    scanlines = [
        b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    ]
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        )
        + _png_chunk(b"IDAT", zlib.compress(b"".join(scanlines), 9))
        + _png_chunk(b"IEND", b"")
    )


def _make_smoke_images(image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    width = 640
    height = 480
    shifts = tuple(range(0, 144, 12))
    canvas_width = width + shifts[-1]
    rng = random.Random(3)
    canvas = bytearray([180] * (canvas_width * height * 3))

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < canvas_width and 0 <= y < height:
            offset = (y * canvas_width + x) * 3
            canvas[offset : offset + 3] = bytes(color)

    for idx in range(450):
        x = rng.randrange(20, canvas_width - 20)
        y = rng.randrange(20, height - 20)
        color = (20, 20, 20) if idx % 2 else (245, 245, 245)
        radius = rng.randrange(3, 9)
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if (xx - x) ** 2 + (yy - y) ** 2 <= radius**2:
                    set_pixel(xx, yy, color)
        inverse = (255 - color[0], 255 - color[1], 255 - color[2])
        for delta in range(-10, 11):
            set_pixel(x + delta, y, inverse)
            set_pixel(x, y + delta, inverse)

    for image_idx, shift in enumerate(shifts, start=1):
        crop = bytearray()
        for y in range(height):
            start = (y * canvas_width + shift) * 3
            crop.extend(canvas[start : start + width * 3])
        _write_png_rgb(
            image_dir / f"image_{image_idx:03d}.png", width, height, crop
        )


def _run(command: list[str], cwd: Path) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % IMAGE_PAIR_ID_MAX
    image_id1 = (pair_id - image_id2) // IMAGE_PAIR_ID_MAX
    return image_id1, image_id2


def _lc_row_count(
    rows: int, is_loop_closure: int, mask_blob: bytes | None
) -> int:
    if mask_blob is None:
        return rows if is_loop_closure else 0
    return sum(1 for value in mask_blob if value)


def _inspect_database(database_path: Path, expected_num_images: int) -> None:
    connection = sqlite3.connect(database_path)
    try:
        image_rows = connection.execute(
            "SELECT image_id, name FROM images ORDER BY name;"
        ).fetchall()
        if len(image_rows) != expected_num_images:
            raise RuntimeError(
                "expected "
                f"{expected_num_images} images, found {len(image_rows)}"
            )

        ordered_image_ids = [image_id for image_id, _ in image_rows]
        image_names = {image_id: name for image_id, name in image_rows}
        image_positions = {
            image_id: position
            for position, image_id in enumerate(ordered_image_ids)
        }
        geometries = {}
        for pair_id, rows, is_loop_closure, mask_blob in connection.execute(
            "SELECT pair_id, rows, is_loop_closure, inlier_matches_are_lc "
            "FROM two_view_geometries WHERE rows > 0;"
        ):
            image_id1, image_id2 = _pair_id_to_image_ids(pair_id)
            lc_rows = _lc_row_count(rows, is_loop_closure, mask_blob)
            distance = abs(
                image_positions[image_id1] - image_positions[image_id2]
            )
            geometries[tuple(sorted((image_id1, image_id2)))] = {
                "distance": distance,
                "rows": rows,
                "is_loop_closure": bool(is_loop_closure),
                "lc_rows": lc_rows,
                "names": (image_names[image_id1], image_names[image_id2]),
            }

        adjacent_failures = []
        for image_id1, image_id2 in zip(
            ordered_image_ids, ordered_image_ids[1:], strict=False
        ):
            geometry = geometries.get(tuple(sorted((image_id1, image_id2))))
            if geometry is None:
                adjacent_failures.append(
                    (image_names[image_id1], image_names[image_id2], "missing")
                )
                continue
            if geometry["is_loop_closure"] or geometry["lc_rows"] != 0:
                adjacent_failures.append(geometry)
        if adjacent_failures:
            raise RuntimeError(
                "adjacent/direct pairs carried LC provenance: "
                f"{adjacent_failures}"
            )

        non_adjacent_lc = [
            geometry
            for geometry in geometries.values()
            if geometry["distance"] > 1
            and geometry["is_loop_closure"]
            and geometry["lc_rows"] > 0
        ]
        if not non_adjacent_lc:
            raise RuntimeError("no verified non-adjacent LC pair found")

        print(
            {
                "adjacent_pairs_checked": len(ordered_image_ids) - 1,
                "non_adjacent_lc_pairs": len(non_adjacent_lc),
                "non_adjacent_lc_rows": sum(
                    geometry["lc_rows"] for geometry in non_adjacent_lc
                ),
            }
        )
    finally:
        connection.close()


def _prepare_work_dir(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    if any(work_dir.iterdir()):
        raise RuntimeError(f"work dir must be empty: {work_dir}")


def run_smoke(colmap_path: Path, work_dir: Path) -> None:
    if not colmap_path.exists():
        raise RuntimeError(f"COLMAP binary does not exist: {colmap_path}")
    if not colmap_path.is_file():
        raise RuntimeError(f"COLMAP path is not a file: {colmap_path}")

    _prepare_work_dir(work_dir)
    image_dir = work_dir / "images"
    sparse_dir = work_dir / "sparse"
    database_path = work_dir / "database.db"
    _make_smoke_images(image_dir)
    sparse_dir.mkdir()

    _run(
        [
            str(colmap_path),
            "feature_extractor",
            "--default_random_seed",
            "1",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.camera_model",
            "SIMPLE_PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--FeatureExtraction.use_gpu",
            "0",
            "--FeatureExtraction.num_threads",
            "1",
            "--SiftExtraction.peak_threshold",
            "0.001",
            "--SiftExtraction.max_num_features",
            "4096",
        ],
        work_dir,
    )
    _run(
        [
            str(colmap_path),
            "sequential_matcher",
            "--default_random_seed",
            "1",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            "0",
            "--FeatureMatching.num_threads",
            "1",
            "--SiftMatching.cpu_brute_force_matcher",
            "1",
            "--SequentialMatching.overlap",
            "3",
            "--SequentialMatching.quadratic_overlap",
            "0",
            "--SequentialMatching.mark_non_consecutive_as_lc",
            "1",
            "--SequentialMatching.mark_loop_detection_as_lc",
            "1",
            "--TwoViewGeometry.min_num_inliers",
            "8",
            "--TwoViewGeometry.multiple_models",
            "0",
            "--TwoViewGeometry.compute_relative_pose",
            "1",
            "--TwoViewGeometry.random_seed",
            "1",
        ],
        work_dir,
    )
    _run(
        [
            str(colmap_path),
            "global_mapper",
            "--default_random_seed",
            "1",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--GlobalMapper.track_lc_second_pass",
            "1",
            "--GlobalMapper.gp_use_lc_observations",
            "1",
            "--GlobalMapper.gp_use_gpu",
            "0",
            "--GlobalMapper.ba_ceres_use_gpu",
            "0",
            "--GlobalMapper.random_seed",
            "1",
            "--GlobalMapper.num_threads",
            "1",
            "--GlobalMapper.min_num_matches",
            "8",
        ],
        work_dir,
    )
    if not (sparse_dir / "0").is_dir():
        raise RuntimeError(
            f"global_mapper did not write a model in {sparse_dir}"
        )
    _inspect_database(database_path, expected_num_images=12)
    print({"database_path": str(database_path), "sparse_path": str(sparse_dir)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--colmap",
        type=Path,
        default=DEFAULT_COLMAP,
        help="Path to the COLMAP CLI binary.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help=(
            "Empty directory for generated smoke-test data. "
            "Defaults to a temp dir."
        ),
    )
    args = parser.parse_args()

    if args.work_dir is not None:
        run_smoke(args.colmap.resolve(), args.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(
            prefix="colmap_native_lc_smoke_"
        ) as tmp:
            run_smoke(args.colmap.resolve(), Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
