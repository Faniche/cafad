"""Audit recorded synthetic injections; these records are not natural gold labels.

The producer sorts each ICAO group by its original time strings before recording
zero-based positions. Callers must preserve that order, including tie order, and
carry a separate per-ICAO offset across stream chunks. Timestamp matches are an
audit comparator and must never replace the exact positional injection mask.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd


class _RecordsUnpickler(pickle.Unpickler):
    """Only NumPy array/scalar reconstruction needed by the observed NPY file."""

    def find_class(self, module, name):
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name in {"_reconstruct", "scalar"}:
                # NumPy 1.x and 2.x pickle spellings resolve to installed NumPy.
                numpy_core = getattr(np, "_core", None)
                if numpy_core is None:  # NumPy 1.x installation.
                    numpy_core = np.core
                return getattr(numpy_core.multiarray, name)
        raise pickle.UnpicklingError("Forbidden pickle global: %s.%s" % (module, name))

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent pickle references are forbidden")


def load_records(path: Path):
    """Load a scalar-object NPY dict without unrestricted ``allow_pickle=True``.

    This is a restricted deserializer, not a general resource-exhaustion sandbox.
    Validation is separate so the original record schema can be inspected first.
    """
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Only NPY versions 1.0 and 2.0 are supported")
        if shape != () or dtype != np.dtype(object):
            raise ValueError("Records must be a scalar object NPY array")
        value = _RecordsUnpickler(stream).load()
        if stream.read(1):
            raise ValueError("Trailing bytes after records payload")
    if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != object:
        raise ValueError("NPY payload does not match its scalar object header")
    records = value.item()
    if not isinstance(records, dict):
        raise ValueError("Records payload must contain a dictionary")
    return records


_TIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[. ]\d{1,9})?$"
)
_NAT = np.iinfo(np.int64).min


def normalize_times(series) -> np.ndarray:
    """Return timezone-naive wall-clock nanoseconds; never infer epoch units.

    Accept ordinary decimal fractions and the producer's space-before-``%f``
    spelling. No timezone is inferred, and invalid/NaT/out-of-range values fail.
    """
    values = np.asarray(series)
    if values.ndim != 1:
        raise ValueError("Timestamps must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind in "iufb":
        raise ValueError("Numeric timestamps require an explicit unit; expected date strings")
    normalized = []
    for item in values:
        if pd.isna(item):
            raise ValueError("NaT or missing timestamp")
        if not isinstance(item, (str, np.str_, np.datetime64, pd.Timestamp)):
            raise ValueError("Unsupported timestamp value type")
        text = str(item)
        if not _TIME_PATTERN.fullmatch(text):
            raise ValueError("Unsupported or invalid timestamp: %r" % text)
        text = text[:19].replace("T", " ") + text[19:].replace(" ", ".")
        normalized.append(text)
    try:
        result = pd.to_datetime(normalized, format="mixed", errors="raise").as_unit("ns").asi8
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid or out-of-range timestamps") from exc
    if np.any(result == _NAT):
        raise ValueError("NaT timestamp")
    return np.asarray(result, dtype=np.int64)


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be a nonnegative integer" % name)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError("%s is outside int64 nonnegative range" % name)
    return int(value)


def _ns_array(values):
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("timestamps_ns must be a one-dimensional integer array")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError("timestamps_ns is outside int64 range")
    array = array.astype(np.int64)
    if np.any(array == _NAT):
        raise ValueError("NaT timestamp")
    return array


def validate_records(records, group_lengths: Optional[Mapping[str, int]] = None):
    """Validate lengths, integer positions, uniqueness per kind and all times.

    Overlap between different injection kinds is allowed. Duplicate positions
    within one kind fail; duplicate timestamps are legitimate observations.
    Pass full group lengths for a final out-of-bounds check. Missing ICAO groups
    in that length mapping fail rather than silently dropping recorded injections.
    """
    if not isinstance(records, dict):
        raise ValueError("Records must be a dictionary")
    result = {}
    for icao, kinds in records.items():
        if not isinstance(icao, (str, np.str_)) or not str(icao):
            raise ValueError("ICAO keys must be nonempty strings preserving leading zeros")
        if not isinstance(kinds, dict):
            raise ValueError("Each ICAO entry must be a dictionary")
        bound = None
        if group_lengths is not None:
            if icao not in group_lengths:
                raise ValueError("Recorded ICAO is absent from group lengths: %s" % icao)
            bound = _nonnegative_integer(group_lengths[icao], "group length")
        result[str(icao)] = {}
        for kind, entry in kinds.items():
            if not isinstance(kind, str) or not kind or not isinstance(entry, dict):
                raise ValueError("Injection kinds must map nonempty names to dictionaries")
            if not entry:
                continue
            if "positions" not in entry or not any(k in entry for k in ("timestamps", "timestamps_ns")):
                raise ValueError("Injection entry requires positions and timestamps")
            positions = np.asarray(entry["positions"])
            if positions.ndim != 1 or (positions.size and positions.dtype.kind not in "iu"):
                raise ValueError("Positions must be a one-dimensional integer array")
            if np.any(positions < 0) or np.any(positions > np.iinfo(np.int64).max):
                raise ValueError("Positions must lie within nonnegative int64 range")
            positions = positions.astype(np.int64)
            if np.unique(positions).size != positions.size:
                raise ValueError("Duplicate positions within an injection kind")
            if bound is not None and np.any(positions >= bound):
                raise ValueError("Recorded position is outside ICAO group length")
            timestamps = (_ns_array(entry["timestamps_ns"]) if "timestamps_ns" in entry
                          else normalize_times(entry["timestamps"]))
            if timestamps.size != positions.size:
                raise ValueError("Position/timestamp lengths differ")
            if "timestamps_ns" in entry and "timestamps" in entry:
                if not np.array_equal(timestamps, normalize_times(entry["timestamps"])):
                    raise ValueError("Raw and normalized timestamps disagree")
            record_type = entry.get("type", kind)
            if not isinstance(record_type, (str, np.str_)) or not str(record_type):
                raise ValueError("Injection type must be a nonempty string")
            result[str(icao)][kind] = {"positions": positions.copy(),
                "timestamps_ns": timestamps.copy(), "type": str(record_type)}
    return result


validate_record = validate_records


def label_group(icao, timestamps_ns, offset, records, total_length=None):
    """Compare exact positions with timestamp joins for one ordered ICAO chunk.

    ``records`` must be the result of :func:`validate_records`. Exact positions
    remain marked even on a time mismatch, which is separately reported as an
    association failure: this function does not certify that mismatched records
    apply to this dataset. Full bounds cannot be inferred from a partial chunk;
    pass ``total_length`` or validate the lengths after finishing the stream.
    """
    times = _ns_array(timestamps_ns)
    offset = _nonnegative_integer(offset, "offset")
    if offset + len(times) > np.iinfo(np.int64).max:
        raise ValueError("Chunk endpoint exceeds int64 range")
    if total_length is not None:
        total_length = _nonnegative_integer(total_length, "total length")
        if offset + len(times) > total_length:
            raise ValueError("Chunk extends beyond total group length")
    exact = np.zeros(len(times), dtype=bool)
    timestamp = exact.copy()
    mismatch_mask = exact.copy()
    types_masks, timestamp_types_masks = {}, {}
    mismatches = 0
    position_count = 0
    for kind, entry in records.get(icao, {}).items():
        positions = entry["positions"]
        recorded_times = entry["timestamps_ns"]
        if total_length is not None and np.any(positions >= total_length):
            raise ValueError("Recorded position is outside ICAO group length")
        selected = (positions >= offset) & (positions < offset + len(times))
        local = positions[selected] - offset
        kind_mask = np.zeros(len(times), dtype=bool)
        kind_mask[local] = True
        time_mask = np.isin(times, recorded_times)
        wrong_local = local[times[local] != recorded_times[selected]]
        mismatch_mask[wrong_local] = True
        mismatches += len(wrong_local)
        position_count += len(local)
        exact |= kind_mask
        timestamp |= time_mask
        types_masks[kind] = kind_mask
        timestamp_types_masks[kind] = time_mask
    return {"exact_mask": exact, "timestamp_mask": timestamp,
            "types_masks": types_masks, "timestamp_types_masks": timestamp_types_masks,
            "position_timestamp_mismatch": int(mismatches),
            "mismatch_positions_mask": mismatch_mask,
            "record_positions_in_chunk": int(position_count),
            "timestamp_only_matches": int(np.count_nonzero(timestamp & ~exact))}
