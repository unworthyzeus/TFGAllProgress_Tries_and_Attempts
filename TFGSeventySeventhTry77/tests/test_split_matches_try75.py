"""Verify Try 77's city-holdout split matches Try 75 byte-for-byte.

Loads ``_split_hdf5_samples`` from Try 75 directly and compares it against
``src.data_utils.split_city_holdout``. Runs on the real CKM HDF5 when
available, otherwise on a deterministic synthetic reference list.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

TFGPRACTICE = Path(__file__).resolve().parents[2]
TRY75_DATA_UTILS = TFGPRACTICE / "TFGSeventyFifthTry75" / "data_utils.py"
TRY77_ROOT = Path(__file__).resolve().parents[1]

if str(TRY77_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY77_ROOT))


def _load_try75_split():
    if not TRY75_DATA_UTILS.is_file():
        pytest.skip("Try 75 data_utils not available")
    spec = importlib.util.spec_from_file_location("try75_data_utils", str(TRY75_DATA_UTILS))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        pytest.skip(f"Could not import Try 75 data_utils: {exc}")
    return module._split_hdf5_samples


@pytest.fixture
def synthetic_refs():
    return [(f"city{i:02d}", f"s{j:03d}") for i in range(20) for j in range(10)]


@pytest.mark.parametrize(
    "val_ratio, test_ratio, seed",
    [
        (0.15, 0.15, 42),
        (0.10, 0.20, 7),
        (0.25, 0.25, 13),
        (0.0, 0.15, 42),
        (0.15, 0.0, 42),
    ],
)
def test_synthetic_split_matches(synthetic_refs, val_ratio, test_ratio, seed):
    from src.data_utils import split_city_holdout

    try75_split = _load_try75_split()
    t_tr, t_va, t_te = try75_split(
        synthetic_refs,
        val_ratio=val_ratio,
        split_seed=seed,
        test_ratio=test_ratio,
        split_mode="city_holdout",
    )
    s_tr, s_va, s_te = split_city_holdout(
        synthetic_refs,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=seed,
    )
    assert s_tr == t_tr, "train split diverges from Try 75"
    assert s_va == t_va, "val split diverges from Try 75"
    assert s_te == t_te, "test split diverges from Try 75"


def test_real_hdf5_split_matches():
    hdf5_env = os.environ.get("CKM_HDF5_PATH")
    candidate = Path(hdf5_env) if hdf5_env else TFGPRACTICE / "Datasets" / "CKM_Dataset_270326.h5"
    if not candidate.is_file():
        pytest.skip(f"HDF5 not found at {candidate}")

    import h5py

    from src.data_utils import split_city_holdout

    refs: list[tuple[str, str]] = []
    with h5py.File(str(candidate), "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                refs.append((city, sample))

    try75_split = _load_try75_split()
    t_tr, t_va, t_te = try75_split(
        refs,
        val_ratio=0.15,
        split_seed=42,
        test_ratio=0.15,
        split_mode="city_holdout",
    )
    s_tr, s_va, s_te = split_city_holdout(refs, 0.15, 0.15, 42)
    assert s_tr == t_tr
    assert s_va == t_va
    assert s_te == t_te
