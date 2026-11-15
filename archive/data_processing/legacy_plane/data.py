"""Prepare observed CSV trajectories without fabricating labels or timestamps.

Flight-level splitting precedes phase segmentation. Standardization and DBSCAN
are fitted on training observations only. Context retrieval never accesses
labels and never borrows validation/test trajectories. The phase rule is an
explicit vertical-rate approximation, not an implementation of Sun et al.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split


FEATURE_NAMES = ["longitude", "latitude", "horizontal_altitude", "pressure_altitude", "speed", "heading_radians"]
ALIASES = {
    "icao": ["icao号", "icao", "icao24", "aircraft_id"],
    "call": ["call号", "call", "callsign"],
    "time": ["时间", "time", "timestamp"],
    "lon": ["经度", "longitude", "lon"],
    "lat": ["纬度", "latitude", "lat"],
    "alt": ["水平高度", "horizontal_altitude", "altitude"],
    "pressure": ["气压高度", "pressure_altitude"],
    "speed": ["速度", "speed"],
    "heading": ["航向", "heading"],
    "label": ["labels", "label", "标签"],
}
NUMERIC = ["lon", "lat", "alt", "pressure", "speed", "heading"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source(path: Path) -> Tuple[pd.DataFrame, dict]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    selected = {}
    for logical, aliases in ALIASES.items():
        matches = [value for value in aliases if value in columns]
        if len(matches) != 1:
            raise ValueError("Expected one unambiguous column for %s; found %s" % (logical, matches))
        selected[logical] = matches[0]
    dtypes = {selected[key]: np.float32 for key in NUMERIC}
    dtypes.update({selected["icao"]: "category", selected["call"]: "category", selected["label"]: np.float32})
    frame = pd.read_csv(path, usecols=list(selected.values()), dtype=dtypes)
    frame = frame.rename(columns={original: logical for logical, original in selected.items()})
    input_rows = len(frame)
    labels = frame["label"].to_numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Input labels must be finite observed binary values; no clipping or replacement is allowed")
    frame["label"] = frame["label"].astype(np.int8)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True, format="mixed")
    invalid_features = ~np.isfinite(frame[NUMERIC].to_numpy()).all(axis=1)
    invalid_time = frame["time"].isna().to_numpy()
    # A callsign is metadata; its absence or change does not define a new aircraft.
    invalid_ids = frame["icao"].isna().to_numpy()
    missing_calls = int(frame["call"].isna().sum())
    if "" not in frame["call"].cat.categories:
        frame["call"] = frame["call"].cat.add_categories([""])
    frame["call"] = frame["call"].fillna("")
    invalid = invalid_features | invalid_time | invalid_ids
    # Do not remove physically unusual, yet finite observations: they may be anomalies.
    frame = frame.loc[~invalid].copy()
    if frame.empty:
        raise ValueError("No finite observations with usable IDs and actual timestamps remain")
    # The documented -1 sentinel must not contaminate a duplicate's valid angle.
    # Other finite physical values are retained; they are not label-conditioned.
    frame["heading"] = frame["heading"].mask(frame["heading"] == -1)
    before_deduplication = len(frame)
    frame = frame.groupby(["icao", "time"], observed=True, sort=False, as_index=False).agg(
        {**{key: "mean" for key in NUMERIC}, "label": "max", "call": "first"}
    )
    frame = frame.sort_values(["icao", "time"], kind="mergesort").reset_index(drop=True)
    return frame, {
        "source_rows": input_rows,
        "dropped_nonfinite_or_invalid_rows": int(invalid.sum()),
        "invalid_feature_rows": int(invalid_features.sum()),
        "invalid_timestamp_rows": int(invalid_time.sum()),
        "missing_identifier_rows": int(invalid_ids.sum()),
        "missing_callsign_metadata_rows_retained": missing_calls,
        "duplicate_timestamp_rows_aggregated": int(before_deduplication - len(frame)),
        "duplicate_aggregation": "Same ICAO/timestamp, even across changed callsigns: arithmetic mean of six measured features, maximum observed label, first callsign value as metadata only",
        "heading_missing_rule": "The documented -1 heading sentinel denotes unavailable heading; ignore it in duplicate means, forward/backward fill within its flight only, otherwise exclude that flight explicitly",
        "finite_physical_outliers_removed": 0,
    }


def _runs(values: np.ndarray) -> List[Tuple[int, int]]:
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))


def _sample_observed_times(times: np.ndarray, count: int) -> np.ndarray:
    """Choose actual observations nearest equally spaced physical times."""
    if count >= len(times):
        return np.arange(len(times), dtype=np.int64)
    targets = np.linspace(times[0], times[-1], count)
    right = np.minimum(np.searchsorted(times, targets), len(times)-1)
    left = np.maximum(right-1, 0)
    selected = np.where(np.abs(times[right]-targets) < np.abs(times[left]-targets), right, left)
    return np.unique(selected).astype(np.int64)


def _phase_ranges(time_seconds: np.ndarray, altitude: np.ndarray, threshold: float) -> List[Tuple[int, int, int]]:
    dt = np.diff(time_seconds)
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Phase segmentation requires strictly increasing observed timestamps")
    vertical = np.r_[0.0, np.diff(altitude.astype(np.float64)) / dt]
    vertical = pd.Series(vertical).rolling(5, min_periods=1, center=True).median().to_numpy()
    phase = np.where(vertical > threshold, 1, np.where(vertical < -threshold, 2, 0)).astype(np.int8)
    # Merge short local runs into a neighboring phase. Never insert measurements.
    while True:
        runs = _runs(phase)
        if len(runs) == 1:
            break
        short = next(((j, start, end) for j, (start, end) in enumerate(runs)
                      if end - start < 3 or time_seconds[end - 1] - time_seconds[start] < 5.0), None)
        if short is None:
            break
        j, start, end = short
        if j == 0:
            neighbor = phase[runs[j + 1][0]]
        elif j == len(runs) - 1:
            neighbor = phase[runs[j - 1][0]]
        else:
            left_duration = time_seconds[runs[j - 1][1] - 1] - time_seconds[runs[j - 1][0]]
            right_duration = time_seconds[runs[j + 1][1] - 1] - time_seconds[runs[j + 1][0]]
            neighbor = phase[runs[j - 1][0]] if left_duration >= right_duration else phase[runs[j + 1][0]]
        phase[start:end] = neighbor
    return [(start, end, int(phase[start])) for start, end in _runs(phase)]


def _wrap_heading(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Paper distance between anchors [absolute seconds, projected metres x/y, heading rad]."""
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt((delta[..., 0] / 10.0) ** 2 + np.sum((delta[..., 1:3] / 10000.0) ** 2, axis=-1)
                   + (_wrap_heading(delta[..., 3]) / np.pi) ** 2)


def _anchor_clusters(anchors: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Sparse exact-metric DBSCAN, using chord embedding only to find candidates."""
    if not len(anchors):
        return np.empty(0, dtype=np.int64)
    embedded = np.column_stack((anchors[:, 0] / 10.0, anchors[:, 1:3] / 10000.0,
                                np.cos(anchors[:, 3]) / np.pi, np.sin(anchors[:, 3]) / np.pi))
    pairs = cKDTree(embedded).query_pairs(eps, output_type="ndarray")
    if len(pairs):
        distances = _distance(anchors[pairs[:, 0]], anchors[pairs[:, 1]])
        select = distances <= eps
        pairs, distances = pairs[select], distances[select]
        # Nonzero epsilon preserves coincident observations in sparse neighborhoods.
        distances = np.maximum(distances, np.finfo(np.float64).eps)
        row = np.r_[pairs[:, 0], pairs[:, 1], np.arange(len(anchors))]
        col = np.r_[pairs[:, 1], pairs[:, 0], np.arange(len(anchors))]
        data = np.r_[distances, distances, np.zeros(len(anchors))]
    else:
        row = col = np.arange(len(anchors))
        data = np.zeros(len(anchors))
    graph = coo_matrix((data, (row, col)), shape=(len(anchors), len(anchors))).tocsr()
    # Recent sklearn can sort by sparse distance values without densifying.
    try:
        from sklearn.neighbors import sort_graph_by_row_values
        graph = sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=False)
    except ImportError:
        pass
    return DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(graph)


def _context_peers(anchors: np.ndarray, phases: np.ndarray, flight_index: np.ndarray,
                   split: np.ndarray, candidates: int, peers_per_context: int,
                   eps: float, min_samples: int,
                   excluded_group_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """Build training-only context sets without reading or filtering any labels."""
    training_segments = np.flatnonzero(split[flight_index] == "train")
    exclusion_groups = flight_index if excluded_group_ids is None else np.asarray(excluded_group_ids)
    if exclusion_groups.shape != flight_index.shape:
        raise ValueError("One exclusion group is required per segment")
    peers = np.full((len(anchors), candidates, peers_per_context), -1, dtype=np.int64)
    metadata = {"cluster_counts_by_phase": {}, "training_anchor_noise_by_phase": {},
                "coarse_context_fallback_candidates": 0, "repeated_coarse_candidates": 0,
                "phase_consistent_training_only": True}
    for phase in np.unique(phases):
        pool = training_segments[phases[training_segments] == phase]
        queries = np.flatnonzero(phases == phase)
        if not len(pool):
            raise ValueError("Phase %s has no training peer context; cannot use held-out flights as substitutes" % phase)
        flattened = anchors[pool].reshape(-1, 4)
        owner = np.repeat(pool, anchors.shape[1])
        cluster_labels = _anchor_clusters(flattened, eps, min_samples)
        contexts = []
        centers = []
        for cluster in np.unique(cluster_labels[cluster_labels >= 0]):
            members = np.unique(owner[cluster_labels == cluster])
            context_anchors = flattened[cluster_labels == cluster]
            center = context_anchors.mean(axis=0)
            center[3] = np.arctan2(np.sin(context_anchors[:, 3]).mean(), np.cos(context_anchors[:, 3]).mean())
            contexts.append(members)
            centers.append(center)
        metadata["cluster_counts_by_phase"][str(int(phase))] = len(contexts)
        metadata["training_anchor_noise_by_phase"][str(int(phase))] = int(np.sum(cluster_labels < 0))
        centers = np.asarray(centers)
        for query in queries:
            other = pool[exclusion_groups[pool] != exclusion_groups[query]]
            if not len(other):
                raise ValueError("Phase %s has no other-flight training peer for flight %s; collect more data or revise the declared phase rule" % (phase, flight_index[query]))
            if len(centers):
                order = np.argsort(_distance(anchors[query, :, None, :], centers[None, :, :]).min(axis=0), kind="stable")
            else:
                order = np.empty(0, dtype=np.int64)
            chosen_groups = []
            for context in order:
                valid = contexts[context][exclusion_groups[contexts[context]] != exclusion_groups[query]]
                if len(valid):
                    chosen_groups.append(valid)
                if len(chosen_groups) == candidates:
                    break
            fallback = candidates - len(chosen_groups)
            metadata["coarse_context_fallback_candidates"] += fallback
            metadata["repeated_coarse_candidates"] += max(0, fallback - 1)
            chosen_groups.extend([other] * fallback)
            for candidate, members in enumerate(chosen_groups):
                # Nearest peer segments by minimum exact anchor-to-anchor distance.
                distances = _distance(anchors[query, None, :, None, :], anchors[members, None, :, :]).min(axis=(1, 2))
                ranked = members[np.argsort(distances, kind="stable")]
                seen_flights = set()
                selected = []
                for member in ranked:
                    peer_flight = str(exclusion_groups[member])
                    if peer_flight not in seen_flights:
                        selected.append(int(member))
                        seen_flights.add(peer_flight)
                    if len(selected) == peers_per_context:
                        break
                peers[query, candidate, :len(selected)] = selected
    valid = peers >= 0
    if not np.all(valid.any(axis=-1)):
        raise AssertionError("Empty context candidate must never be replaced by a synthetic peer")
    if np.any(split[flight_index[peers[valid]]] != "train"):
        raise AssertionError("Held-out observations leaked into context")
    target_flights = np.broadcast_to(flight_index[:, None, None], peers.shape)
    if np.any(flight_index[peers[valid]] == target_flights[valid]):
        raise AssertionError("Target flight leaked into its own peer context")
    target_groups = np.broadcast_to(exclusion_groups[:, None, None], peers.shape)
    if np.any(exclusion_groups[peers[valid]] == target_groups[valid]):
        raise AssertionError("Target aircraft identity leaked into its own peer context")
    metadata["train_peer_only_assertion"] = True
    metadata["other_flight_only_assertion"] = True
    metadata["other_aircraft_only_assertion"] = excluded_group_ids is not None
    return peers, metadata


def prepare_data(source: Path, output: Path, max_points: int = 32, max_flights: int = 0,
                 split_seed: int = 42, gap_seconds: float = 1800.0,
                 phase_vertical_threshold: float = 1.0, anchors_per_segment: int = 3,
                 dbscan_eps: float = 1.0, dbscan_min_samples: int = 3,
                 context_candidates: int = 2, peers_per_context: int = 3,
                 min_points: int = 8) -> dict:
    """Prepare observed CSV trajectories and persist a complete split/context audit."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_points < 2 or min_points < 2 or max_flights < 0 or gap_seconds <= 0 or phase_vertical_threshold <= 0:
        raise ValueError("Invalid point, flight, gap or phase settings")
    if min(context_candidates, peers_per_context, anchors_per_segment, dbscan_min_samples) < 1 or dbscan_eps <= 0:
        raise ValueError("Context counts, epsilon and min_samples must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepared output must be new or empty; refusing to overwrite provenance: %s" % output)
    print("Reading actual source observations: %s" % source, flush=True)
    frame, counts = _read_source(source)
    timestamps = frame["time"].astype("int64").to_numpy()
    aircraft = frame["icao"].cat.codes.to_numpy()
    day = timestamps // (86400 * 10**9)
    boundary = np.r_[True, (aircraft[1:] != aircraft[:-1])
                     | (day[1:] != day[:-1]) | (np.diff(timestamps) > gap_seconds * 1e9)]
    cuts = np.r_[np.flatnonzero(boundary), len(frame)]
    flight_records = []
    flight_slices = []
    excluded = []
    heading_filled_points = 0
    for start, end in zip(cuts[:-1], cuts[1:]):
        part = frame.iloc[start:end]
        row = part.iloc[0]
        flight_id = "%s|%s" % (row["icao"], int(timestamps[start]))
        if end - start < min_points:
            excluded.append({"flight_id": flight_id, "reason": "fewer_than_min_points", "points": int(end-start)})
            continue
        heading = part["heading"].ffill().bfill()
        if heading.isna().any():
            excluded.append({"flight_id": flight_id, "reason": "heading_unavailable_for_entire_flight", "points": int(end-start)})
            continue
        missing = part["heading"].isna().to_numpy()
        heading_filled_points += int(missing.sum())
        frame.loc[start:end-1, "heading"] = heading.to_numpy()
        flight_records.append({"flight_id": flight_id, "icao": str(row["icao"]), "call": str(row["call"]),
                               "start_time_utc": str(part["time"].iloc[0]), "end_time_utc": str(part["time"].iloc[-1]),
                               "label": int(part["label"].max()), "points": int(end-start)})
        flight_slices.append((int(start), int(end)))
    if max_flights and len(flight_records) > max_flights:
        selected = sorted(range(len(flight_records)), key=lambda j: hashlib.sha256(flight_records[j]["flight_id"].encode()).hexdigest())[:max_flights]
        selected.sort()
        flight_records = [flight_records[j] for j in selected]
        flight_slices = [flight_slices[j] for j in selected]
    labels = np.asarray([record["label"] for record in flight_records], dtype=np.int64)
    if len(labels) < 10 or len(np.unique(labels)) != 2:
        raise ValueError("At least ten observed flights with both labels are required for an independent stratified train/validation/test split")
    flight_icao = np.asarray([record["icao"] for record in flight_records])
    aircraft_ids, aircraft_index = np.unique(flight_icao, return_inverse=True)
    aircraft_labels = np.zeros(len(aircraft_ids), dtype=np.int64)
    np.maximum.at(aircraft_labels, aircraft_index, labels)
    try:
        group_train, held_out = train_test_split(np.arange(len(aircraft_ids)), train_size=0.7, stratify=aircraft_labels, random_state=split_seed)
        group_validation, group_test = train_test_split(held_out, test_size=0.5, stratify=aircraft_labels[held_out], random_state=split_seed)
    except ValueError as error:
        raise ValueError("Insufficient observed flights per class for the declared 70/15/15 split: %s" % error) from error
    group_split = np.full(len(aircraft_ids), "train", dtype="U5")
    group_split[group_validation], group_split[group_test] = "val", "test"
    split = group_split[aircraft_index]
    train, validation, test = [np.flatnonzero(split == name) for name in ("train", "val", "test")]
    for subset in (train, validation, test):
        if len(np.unique(labels[subset])) != 2:
            raise ValueError("Every independent split must contain both observed labels; collect more flights")
    for index, record in enumerate(flight_records):
        record.update({"flight_index": index, "split": str(split[index])})
    absolute_origin = int(timestamps.min())
    features, elapsed, masks, segment_flights, phases, raw_anchors = [], [], [], [], [], []
    segment_records = []
    for index, (start, end) in enumerate(flight_slices):
        part = frame.iloc[start:end]
        times = (timestamps[start:end] - absolute_origin).astype(np.float64) / 1e9
        values = part[NUMERIC].to_numpy(dtype=np.float64)
        values[:, 5] = np.deg2rad(values[:, 5] % 360.0)
        for phase_start, phase_end, phase in _phase_ranges(times, values[:, 2], phase_vertical_threshold):
            length = phase_end-phase_start
            if length < 2:
                raise ValueError("Phase approximation produced fewer than two observed timestamps")
            sample = phase_start + _sample_observed_times(times[phase_start:phase_end], min(max_points, length))
            # Repeated anchors are allowed for a very short real phase; target points remain unique.
            anchor_targets = np.linspace(times[phase_start], times[phase_end-1], anchors_per_segment)
            relative_times = times[phase_start:phase_end]
            right = np.minimum(np.searchsorted(relative_times, anchor_targets), length-1)
            left = np.maximum(right-1, 0)
            anchor_sample = phase_start + np.where(np.abs(relative_times[right]-anchor_targets) < np.abs(relative_times[left]-anchor_targets), right, left)
            observed = len(sample)
            f = np.zeros((max_points, 6), dtype=np.float32)
            t = np.zeros(max_points, dtype=np.float32)
            mask = np.zeros(max_points, dtype=bool)
            f[:observed] = values[sample]
            t[:observed] = times[sample] - times[sample[0]]
            mask[:observed] = True
            features.append(f)
            elapsed.append(t)
            masks.append(mask)
            segment_flights.append(index)
            phases.append(phase)
            # Projection occurs below, fitted on train positions only.
            raw_anchors.append(np.column_stack((times[anchor_sample], values[anchor_sample, 0], values[anchor_sample, 1], values[anchor_sample, 5])))
            segment_records.append({"segment_index": len(features)-1, "flight_index": index,
                                    "phase": phase, "start_relative_seconds": float(times[phase_start]),
                                    "end_relative_seconds": float(times[phase_end-1]), "observed_points": length,
                                    "sampled_points": observed})
    features = np.stack(features)
    elapsed = np.stack(elapsed)
    mask = np.stack(masks)
    flight_index = np.asarray(segment_flights, dtype=np.int64)
    phases = np.asarray(phases, dtype=np.int64)
    training_mask = (split[flight_index] == "train")[:, None] & mask
    fitting = features[training_mask].astype(np.float64)
    mean, scale = np.zeros(6), np.ones(6)
    mean[:5] = fitting[:, :5].mean(axis=0)
    scale[:5] = fitting[:, :5].std(axis=0)
    scale[scale < 1e-8] = 1.0
    projected_anchors = np.stack(raw_anchors)
    latitude_reference = np.deg2rad(float(fitting[:, 1].mean()))
    longitude_reference = float(fitting[:, 0].mean())
    projected_anchors[:, :, 1] = 6371000.0 * np.cos(latitude_reference) * np.deg2rad(projected_anchors[:, :, 1] - longitude_reference)
    projected_anchors[:, :, 2] = 6371000.0 * np.deg2rad(projected_anchors[:, :, 2] - np.rad2deg(latitude_reference))
    print("Building phase-consistent contexts using training flights only", flush=True)
    peers, context_info = _context_peers(projected_anchors, phases, flight_index, split,
                                         context_candidates, peers_per_context, dbscan_eps, dbscan_min_samples,
                                         excluded_group_ids=flight_icao[flight_index])
    features[mask] = ((features[mask].astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(elapsed).all():
        raise FloatingPointError("Prepared observations must remain finite")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "data.npz", features=features, elapsed=elapsed, mask=mask,
                        flight_index=flight_index, labels=labels,
                        flight_ids=np.asarray([r["flight_id"] for r in flight_records]), flight_icao=flight_icao, split=split, phases=phases,
                        peers=peers)
    pd.DataFrame(flight_records).to_csv(output / "split.csv", index=False)
    pd.DataFrame(segment_records).to_csv(output / "segments.csv", index=False)
    manifest = {
        "schema_version": 3, "source": str(source), "source_sha256": _sha256(source),
        "data_kind": "supplied_raw_csv_observations", "external_provenance_verified": False,
        "feature_names": FEATURE_NAMES,
        "config": {"max_points": max_points, "max_flights": max_flights, "split_seed": split_seed,
                   "gap_seconds": gap_seconds, "phase_vertical_threshold": phase_vertical_threshold,
                   "anchors_per_segment": anchors_per_segment, "dbscan_eps": dbscan_eps,
                   "dbscan_min_samples": dbscan_min_samples, "context_candidates": context_candidates,
                   "peers_per_context": peers_per_context, "min_points": min_points},
        "flight_definition": "Sort by ICAO and actual timestamp; start a new observed session only on ICAO change, UTC date boundary, or consecutive observed gap greater than gap_seconds. Callsign is metadata only; its change cannot split a continuous physical trajectory. Source sessions can be local observation slices, not guaranteed complete flights.",
        "flight_label": "maximum observed point label over the whole retained flight, before phase segmentation or point subsampling",
        "split_method": "one ICAO-group-stratified 70/15/15 split, using max observed session label per ICAO only for stratification; all sessions of one aircraft stay in one split. Flight ratios may differ slightly. Same fixed split for every training seed.",
        "split_sha256": _sha256(output / "split.csv"),
        "data_sha256": _sha256(output / "data.npz"),
        "segments_sha256": _sha256(output / "segments.csv"),
        "pilot_selection": "smallest stable SHA256 of flight ID, independent of label; zero max_flights uses all eligible flights",
        "phase_method": {"name": "vertical_rate_threshold_approximation", "not_exact_Sun_method": True,
                         "vertical_rate": "difference of observed horizontal altitude over actual elapsed seconds",
                         "smoothing": "centered rolling median, five observed points", "minimum_phase_seconds": 5.0,
                         "minimum_phase_points": 3, "phase_codes": {"0": "level", "1": "climb", "2": "descent"},
                         "short_runs": "iteratively merge into the longer adjacent run; do not fabricate observations"},
        "point_sampling": "actual observations nearest uniformly spaced physical times within each phase; keep original timestamps, deduplicate selected observations, no interpolation or synthetic time steps",
        "scaler": {"fit_split": "train", "fit_count": int(training_mask.sum()), "mean": mean.tolist(), "scale": scale.tolist(),
                   "heading": "kept in radians, unstandardized"},
        "context": {**context_info, "distance": "sqrt((dt/10)^2 + (projected_metres/10000)^2 + (wrapped_heading/pi)^2)",
                    "dbscan_fit_split": "train", "retrieval_uses_labels": False, "exclude_target_entire_flight": True,
                    "exclude_target_all_icao_sessions": True,
                    "candidate_retrieval": "nearest phase-consistent train cluster centroid to any query anchor; sparse exact-distance DBSCAN uses chord embedding only for candidate neighbor search",
                    "peer_selection": "minimum exact anchor distance, distinct other training ICAO per peer slot; -1 pads unused slots",
                    "fallback": "same-phase coarse training set when clusters are missing or have no other-flight peer; repeat coarse candidate if necessary; fail if no other-flight training peer exists",
                    "projection": {"kind": "local equirectangular", "earth_radius_metres": 6371000.0,
                                   "reference_latitude_train_degrees": float(np.rad2deg(latitude_reference)),
                                   "reference_longitude_train_degrees": longitude_reference},
                    "absolute_time_origin_unix_ns": absolute_origin},
        "counts": {**counts, "heading_filled_points": heading_filled_points, "eligible_before_pilot_selection": len(cuts)-1-len(excluded),
                   "excluded_flights": len(excluded), "retained_flights": len(labels), "retained_segments": len(features),
                   "retained_aircraft": len(aircraft_ids),
                   "aircraft_by_split": {name: int(np.sum(group_split == name)) for name in ["train", "val", "test"]},
                   "flights_by_split": {name: int(np.sum(split == name)) for name in ["train", "val", "test"]},
                   "labels_by_split": {name: {str(label): int(np.sum((split == name) & (labels == label))) for label in [0, 1]} for name in ["train", "val", "test"]}},
        "excluded_flight_records": excluded,
        "source_time_min": str(frame["time"].min()), "source_time_max": str(frame["time"].max()),
        "limitations": ["Phase identification is a disclosed vertical-rate approximation, not exact Sun et al.",
                        "Input annotation authenticity and the paper's dataset identity have not been externally verified.",
                        "The source CSV supplies no timezone. Parsing naive timestamps as UTC is a computational convention, not evidence of the original reporting timezone; within-flight elapsed intervals are unchanged.",
                        "Local observation slices can be partial flights."]
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class PreparedData:
    """Load a prepared observation archive and expose label-free model inputs."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 3:
            raise ValueError("Rebuild prepared data with schema3 ICAO isolation")
        for filename, key in [("split.csv", "split_sha256"), ("data.npz", "data_sha256"), ("segments.csv", "segments_sha256")]:
            if key not in self.manifest:
                raise ValueError("Prepared cache lacks %s; regenerate using the current physical-trajectory pipeline" % key)
            if _sha256(self.directory / filename) != self.manifest[key]:
                raise ValueError("Prepared %s hash does not match the saved manifest" % filename)
        with np.load(self.directory / "data.npz", allow_pickle=False) as archive:
            for key in ["features", "elapsed", "mask", "flight_index", "labels", "flight_ids", "flight_icao", "split", "phases", "peers"]:
                setattr(self, key, archive[key])
        recorded_split = pd.read_csv(self.directory / "split.csv", dtype={"flight_id": str, "icao": str})
        for name, actual in [("flight_id", self.flight_ids), ("label", self.labels), ("split", self.split)]:
            if not np.array_equal(recorded_split[name].to_numpy(), actual):
                raise ValueError("Prepared arrays do not match the recorded flight split: %s" % name)
        if not np.array_equal(recorded_split["icao"].to_numpy(), self.flight_icao):
            raise ValueError("Prepared ICAO identities differ from the recorded split")
        if recorded_split.groupby("icao")["split"].nunique().gt(1).any():
            raise ValueError("Aircraft identity leakage across splits")
        if len(self.features) == 0 or self.features.shape[-1] != 6:
            raise ValueError("Prepared data must contain observed six-feature phase segments")
        if not np.isfinite(self.features).all() or not np.isfinite(self.elapsed).all():
            raise ValueError("Non-finite prepared observations")
        if self.mask.shape != self.features.shape[:2] or not np.all(self.mask.sum(axis=1) >= 2):
            raise ValueError("Each prepared phase needs at least two valid observations")
        if np.any(self.flight_index < 0) or np.any(self.flight_index >= len(self.labels)):
            raise ValueError("Invalid prepared segment-to-flight indices")
        valid = self.peers >= 0
        if self.peers.min() < -1 or np.any(self.peers[valid] >= len(self.features)) or not np.all(valid.any(axis=-1)):
            raise ValueError("Invalid or empty prepared context candidates")
        if np.any(self.split[self.flight_index[self.peers[valid]]] != "train"):
            raise ValueError("Prepared context contains held-out observations")
        own_flight = np.broadcast_to(self.flight_index[:, None, None], self.peers.shape)
        if np.any(self.flight_index[self.peers[valid]] == own_flight[valid]):
            raise ValueError("Prepared context contains its target flight")
        segment_aircraft = self.flight_icao[self.flight_index]
        own_aircraft = np.broadcast_to(segment_aircraft[:, None, None], self.peers.shape)
        if np.any(segment_aircraft[self.peers[valid]] == own_aircraft[valid]):
            raise ValueError("Prepared context contains another session of its target aircraft")

    def get_batch(self, flight_indices: np.ndarray) -> Dict[str, np.ndarray]:
        requested = np.asarray(flight_indices, dtype=np.int64)
        if requested.ndim != 1 or not len(requested) or len(np.unique(requested)) != len(requested):
            raise ValueError("A batch must contain distinct requested flight indices")
        if np.any(requested < 0) or np.any(requested >= len(self.labels)):
            raise IndexError("Requested flight index is out of range")
        local = np.full(len(self.labels), -1, dtype=np.int64)
        local[requested] = np.arange(len(requested))
        selected = np.flatnonzero(local[self.flight_index] >= 0)
        peer_indices = self.peers[selected]
        peer_valid = peer_indices >= 0
        safe_peers = np.maximum(peer_indices, 0)
        return {
            "features": self.features[selected], "elapsed": self.elapsed[selected], "mask": self.mask[selected],
            "peers": np.where(peer_valid[..., None, None], self.features[safe_peers], 0).astype(np.float32),
            "peer_elapsed": np.where(peer_valid[..., None], self.elapsed[safe_peers], 0).astype(np.float32),
            "peer_mask": self.mask[safe_peers] & peer_valid[..., None], "peer_valid": peer_valid,
            "trajectory_indices": local[self.flight_index[selected]], "trajectory_labels": self.labels[requested],
            "segment_indices": selected, "flight_indices": requested,
        }
