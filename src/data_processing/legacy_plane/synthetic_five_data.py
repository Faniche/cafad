"""Five predeclared controlled corruptions of real, full-resolution observations.

The supplied raw label column is never selected or read. The binary target means
new corruption was injected, not that an unmodified parent is naturally normal.
All variants/scenarios retain one identity split. No original file is modified.
"""
from pathlib import Path
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from .data import (ALIASES, NUMERIC, FEATURE_NAMES, _sha256, _phase_ranges,
                   _sample_observed_times, _context_peers)


PREFIX = "synthetic-five-v1"
SCENARIOS = ("D1", "D2", "D3", "D4", "D5")


def generation_protocol(source_sha256):
    """Parent-approved, fixed before any synthetic model fitting/selection."""
    return {
        "schema_version": 1, "protocol_id": PREFIX,
        "source_sha256": source_sha256,
        "approval": "Parent approved the exact five-family proposal before generation on 2026-10-27; no intensity changes based on validation/test performance permitted.",
        "source_attestation": {"provided_by": "user", "claim": "The supplied original raw is unprocessed.",
                               "independently_verified": False},
        "original_labels": "Not read, parsed, validated, used for eligibility, split, noise, features or target. Only new injection defines the target.",
        "negative_definition": "Unmodified supplied observations with no additional injection; not natural-normal gold labels.",
        "split": {"unit": "ICAO", "hash_prefix": PREFIX,
                  "method": "Sort eligible ICAO by SHA256(prefix+'|'+icao); first floor(.70*N) train, next floor(.15*N) val, remainder test; all sessions/variants/scenarios locked.",
                  "train_fraction": .70, "val_fraction": .15, "test_remainder": True},
        "session": {"gap_seconds": 1800., "date_boundary": True, "callsign_boundary": False,
                    "dedup": "Same ICAO and timestamp: mean physical values, first callsign metadata; original labels excluded."},
        "eligibility": {"min_parent_points": 64, "min_parent_seconds": 60.,
                        "min_phase_points": 32, "min_phase_seconds": 20.,
                        "heading": "-1 missing sentinel; within-parent forward/backward fill; exclude fully unavailable parent."},
        "phase": {"name": "offline_vertical_rate_approximation_not_Sun",
                  "vertical_threshold_native_altitude_units_per_second": 1.,
                  "smoothing": "centered rolling median5", "min_run_points": 3, "min_run_seconds": 5.,
                  "short_run_merge": "longer adjacent run",
                  "boundaries": "Computed from full-resolution parent before injection and shared by clean/corrupted pair; controlled paired-window experiment, not causal deployment."},
        "selection": {"phase": "One eligible phase per parent using independent SHA256-derived NumPy generator; independent of original labels and all model results.",
                      "support": "Actual observations in the selected phase's physical-time interval [20%,80%], inclusive.",
                      "wave": "u=(t-support_start)/(support_end-support_start); w=sin(pi*u)^2 inside support; zero outside."},
        "scenarios": {
            "D1": {"family": "speed_gaussian", "sigma_floor": 15., "sigma_abs_speed_factor": .15},
            "D2": {"family": "sustained_speed", "abs_speed_factor": .30, "speed_offset": 20.},
            "D3": {"family": "altitude_wave", "horizontal_amplitude": 350., "pressure_amplitude": 280.},
            "D4": {"family": "lateral_heading_wave", "lateral_metres": 2500., "heading_degrees": 30.},
            "D5": {"family": "mixed", "sigma_floor": 10., "sigma_abs_speed_factor": .10,
                   "abs_speed_factor": .20, "speed_offset": 15., "horizontal_amplitude": 250.,
                   "pressure_amplitude": 200., "lateral_metres": 1800., "heading_degrees": 20.}},
        "numerics": {"earth_radius_metres": 6371000., "speed_floor_after_injection": 0.,
                     "heading": "modulo360 degrees", "coordinates": "fail on invalid coordinates; no clipping",
                     "units": "Speed/altitude amplitudes are native input units; external physical-unit provenance is unverified.",
                     "lateral": "Lat+=cos(direction)*metres*w/R in radians; lon+=sin(direction)*metres*w/(R*cos(parent mean latitude)); direction uniform[0,2pi)."},
        "rng": {"implementation": "numpy.random.default_rng(PCG64)",
                "seed": "First16 bytes of SHA256(prefix+'|'+scope+'|'+parent_id+'|'+scenario) interpreted big-endian integer; phase scope uses empty scenario.",
                "seed_search": False},
        "sampling": {"max_points": 64, "method": "Full-resolution injection first; then nearest actual observations to equally spaced physical times within each pre-injection phase; unique indices; original timestamps."},
        "contexts": {"reference_pool": "Training clean variants only, excluding all target ICAO sessions.",
                     "scaler_projection_dbscan_fit": "train-clean sampled original observations only",
                     "candidates": 2, "peers_per_candidate": 3, "dbscan_eps": 1., "dbscan_min_samples": 3,
                     "anchors": 3, "distance": "sqrt((absolute_dt/10)^2+(projected_distance_metres/10000)^2+(wrap_heading/pi)^2)",
                     "retrieval": "Existing exact-metric phasewise DBSCAN and nearest centroid/context, same-phase coarse fallback; not historical precedence.",
                     "paper_difference": "Reference pool restricted to unmodified training parents rather than all training examples; explicit controlled-corruption protocol adaptation."},
        "balance": "Exactly one clean(label0) and one corrupted(label1) per eligible parent per scenario, same split.",
        "limits": ["D1-D5 are five constructed scenarios, never five observed months or proof of old 5-9 datasets.",
                   "Original dates are retained; naive source timezone is unknown (UTC parsing is computational only).",
                   "Offline phase boundaries and heading filling are not streaming causal preprocessing.",
                   "No accuracy, precision, recall or F1 guarantee; held-out performance must be computed honestly."]}


def _write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def freeze_protocol(source, output):
    output = Path(output)
    protocol = generation_protocol(_sha256(Path(source)))
    _write_new_json(output / "generation_protocol.json", protocol)
    return protocol


def _seed(scope, parent_id, scenario=""):
    text = "|".join((PREFIX, scope, str(parent_id), scenario))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big"), digest.hex()


def _read_unlabelled_source(source):
    header = pd.read_csv(source, nrows=0).columns.tolist()
    selected = {}
    for logical in ("icao", "call", "time", *NUMERIC):
        found = [name for name in ALIASES[logical] if name in header]
        if len(found) != 1:
            raise ValueError("Expected one source column for %s" % logical)
        selected[logical] = found[0]
    dtypes = {selected[key]: np.float64 for key in NUMERIC}
    dtypes.update({selected["icao"]: "category", selected["call"]: "category"})
    frame = pd.read_csv(source, usecols=list(selected.values()), dtype=dtypes)
    frame = frame.rename(columns={value: key for key, value in selected.items()})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True, format="mixed")
    invalid = (~np.isfinite(frame[NUMERIC].to_numpy()).all(axis=1)
               | frame["time"].isna().to_numpy() | frame["icao"].isna().to_numpy())
    counts = {"source_rows": len(frame), "dropped_nonfinite_time_or_id": int(invalid.sum()),
              "original_label_column_read": False, "finite_outliers_removed": 0}
    frame = frame.loc[~invalid].copy()
    frame["heading"] = frame["heading"].mask(frame["heading"] == -1)
    before = len(frame)
    frame = frame.groupby(["icao", "time"], observed=True, sort=False, as_index=False).agg(
        {**{key: "mean" for key in NUMERIC}, "call": "first"})
    frame = frame.sort_values(["icao", "time"], kind="mergesort").reset_index(drop=True)
    counts.update({"duplicate_rows_aggregated": before-len(frame), "deduplicated_rows": len(frame)})
    if not len(frame):
        raise ValueError("No valid original observations")
    return frame, counts


def _parents(frame, protocol):
    timestamps = frame["time"].astype("int64").to_numpy()
    codes = frame["icao"].cat.codes.to_numpy()
    day = timestamps // (86400 * 10**9)
    boundary = np.r_[True, (codes[1:] != codes[:-1]) | (day[1:] != day[:-1])
                     | (np.diff(timestamps) > protocol["session"]["gap_seconds"] * 1e9)]
    cuts = np.r_[np.flatnonzero(boundary), len(frame)]
    parents, exclusions = [], []
    origin = int(timestamps.min())
    config = protocol["eligibility"]
    for start, end in zip(cuts[:-1], cuts[1:]):
        part = frame.iloc[start:end]
        parent_id = "%s|%d" % (part["icao"].iloc[0], timestamps[start])
        reason = None
        duration = (timestamps[end-1]-timestamps[start]) / 1e9
        if end-start < config["min_parent_points"]:
            reason = "insufficient_parent_points"
        elif duration < config["min_parent_seconds"]:
            reason = "insufficient_parent_duration"
        heading = part["heading"].ffill().bfill()
        if reason is None and heading.isna().any():
            reason = "heading_unavailable"
        if reason:
            exclusions.append({"parent_id": parent_id, "reason": reason})
            continue
        times = (timestamps[start:end]-origin).astype(np.float64) / 1e9
        values = part[NUMERIC].to_numpy(dtype=np.float64, copy=True)
        values[:, 5] = heading.to_numpy() % 360.
        ranges = _phase_ranges(times, values[:, 2], protocol["phase"]["vertical_threshold_native_altitude_units_per_second"])
        eligible = [i for i, (a, b, _) in enumerate(ranges)
                    if b-a >= config["min_phase_points"] and times[b-1]-times[a] >= config["min_phase_seconds"]]
        if not eligible:
            exclusions.append({"parent_id": parent_id, "reason": "no_eligible_phase"})
            continue
        seed, digest = _seed("phase", parent_id)
        chosen = int(np.random.default_rng(seed).choice(eligible))
        a, b, _ = ranges[chosen]
        support = (float(times[a]+.2*(times[b-1]-times[a])), float(times[a]+.8*(times[b-1]-times[a])))
        parents.append({"parent_id": parent_id, "icao": str(part["icao"].iloc[0]), "call": str(part["call"].iloc[0]),
                        "times": times, "values": values, "ranges": ranges, "selected_phase": chosen,
                        "support": support, "phase_seed_digest": digest, "heading_filled_points": int(part["heading"].isna().sum()),
                        "start_time_utc": str(part["time"].iloc[0]), "end_time_utc": str(part["time"].iloc[-1]),
                        "start_unix_ns": int(timestamps[start]), "end_unix_ns": int(timestamps[end-1])})
    if len(parents) < 10:
        raise ValueError("At least ten eligible parent sessions required")
    identities = sorted({p["icao"] for p in parents},
                        key=lambda x: hashlib.sha256((PREFIX+"|"+x).encode()).hexdigest())
    n_train, n_val = int(.7*len(identities)), int(.15*len(identities))
    if min(n_train, n_val, len(identities)-n_train-n_val) < 1:
        raise ValueError("Not enough independent ICAO for three splits")
    roles = {icao: ("train" if i < n_train else "val" if i < n_train+n_val else "test")
             for i, icao in enumerate(identities)}
    for parent in parents:
        parent["split"] = roles[parent["icao"]]
    return parents, exclusions, origin


def _corrupt(parent, scenario, protocol):
    values = parent["values"].copy()
    times = parent["times"]
    first, last = parent["support"]
    support = (times >= first) & (times <= last)
    if support.sum() < 2:
        raise ValueError("Full-resolution support contains fewer than two observations")
    u = (times[support]-first)/(last-first)
    wave = np.sin(np.pi*u)**2
    seed, digest = _seed("noise", parent["parent_id"], scenario)
    rng = np.random.default_rng(seed)
    config = protocol["scenarios"][scenario]
    direction = None
    original = parent["values"][support]
    if scenario in ("D1", "D5"):
        sigma = np.maximum(config["sigma_floor"], config["sigma_abs_speed_factor"]*np.abs(original[:, 4]))
        values[support, 4] += rng.normal(0., sigma)
    if scenario in ("D2", "D5"):
        values[support, 4] += (config["abs_speed_factor"]*np.abs(original[:, 4])+config["speed_offset"])*wave
    if scenario in ("D3", "D5"):
        values[support, 2] += config["horizontal_amplitude"]*wave
        values[support, 3] += config["pressure_amplitude"]*wave
    if scenario in ("D4", "D5"):
        direction = float(rng.uniform(0., 2*np.pi))
        distance = config["lateral_metres"]*wave
        radius = protocol["numerics"]["earth_radius_metres"]
        cosine = np.cos(np.deg2rad(parent["values"][:, 1].mean()))
        if abs(cosine) < 1e-6:
            raise ValueError("Local longitude projection unstable near pole")
        values[support, 1] += np.rad2deg(np.cos(direction)*distance/radius)
        values[support, 0] += np.rad2deg(np.sin(direction)*distance/(radius*cosine))
        values[support, 5] = (values[support, 5]+config["heading_degrees"]*wave) % 360.
    values[support, 4] = np.maximum(0., values[support, 4])
    if not np.isfinite(values).all() or np.any(np.abs(values[:, 1]) > 90) or np.any(np.abs(values[:, 0]) > 180):
        raise ValueError("Corruption produced nonfinite or invalid coordinates")
    changed = np.any(values != parent["values"], axis=1)
    if not changed.any():
        raise ValueError("No actual physical values changed")
    if np.any(changed & ~support):
        raise AssertionError("Changes outside declared support")
    ledger = {"parent_id": parent["parent_id"], "scenario": scenario, "rng_seed": str(seed),
              "rng_seed_digest": digest, "selected_phase": parent["selected_phase"],
              "support_start_relative_seconds": first, "support_end_relative_seconds": last,
              "support_point_indices": np.flatnonzero(support).tolist(),
              "actually_changed_point_indices": np.flatnonzero(changed).tolist(),
              "full_parent_points": len(values), "injection_support_points": int(support.sum()),
              "actual_changed_points": int(changed.sum()), "direction_radians": direction,
              "changed_values_sha256": hashlib.sha256(np.ascontiguousarray(values[changed], dtype="<f8").tobytes()).hexdigest(),
              "original_changed_values_sha256": hashlib.sha256(np.ascontiguousarray(parent["values"][changed], dtype="<f8").tobytes()).hexdigest(),
              "parameters": config}
    return values, changed, ledger


def _prepare_scenario(parents, scenario, protocol, output, source, origin, source_counts,
                      protocol_path, parent_split_path, cohort_path, addendum_path):
    if output.exists():
        raise FileExistsError("Scenario directory must not exist: %s" % output)
    maximum = protocol["sampling"]["max_points"]
    features, elapsed, masks, owners, phases, anchors, segment_labels = [], [], [], [], [], [], []
    flights, segments, ledgers = [], [], []
    for parent in parents:
        corrupted, changed, ledger = _corrupt(parent, scenario, protocol)
        ledgers.append(ledger)
        for variant, values, point_labels in (("clean", parent["values"], np.zeros(len(changed), dtype=bool)),
                                               ("corrupted", corrupted, changed)):
            flight_index = len(flights)
            flights.append({"flight_index": flight_index, "flight_id": parent["parent_id"]+"|"+variant,
                            "parent_id": parent["parent_id"], "icao": parent["icao"], "call": parent["call"],
                            "variant": variant, "label": int(variant == "corrupted"), "split": parent["split"],
                            "points": len(values), "start_time_utc": parent["start_time_utc"], "end_time_utc": parent["end_time_utc"]})
            times = parent["times"]
            for ordinal, (start, end, phase) in enumerate(parent["ranges"]):
                if end-start < 2:
                    raise ValueError("Parent phase has fewer than two observations")
                indices = start + _sample_observed_times(times[start:end], min(maximum, end-start))
                observed = len(indices)
                f, t, mask = np.zeros((maximum, 6), np.float32), np.zeros(maximum, np.float32), np.zeros(maximum, bool)
                f[:observed] = values[indices]
                f[:observed, 5] = np.deg2rad(values[indices, 5])
                t[:observed] = times[indices]-times[indices[0]]
                mask[:observed] = True
                anchor_indices = start + _sample_observed_times(times[start:end], 3)
                # Very short real phases may need repeated anchors only, not inserted model observations.
                if len(anchor_indices) < 3:
                    anchor_indices = np.resize(anchor_indices, 3)
                features.append(f); elapsed.append(t); masks.append(mask); owners.append(flight_index); phases.append(phase)
                anchors.append(np.column_stack((times[anchor_indices], values[anchor_indices, 0], values[anchor_indices, 1], np.deg2rad(values[anchor_indices, 5]))))
                label = int(point_labels[start:end].any())
                segment_labels.append(label)
                segments.append({"segment_index": len(features)-1, "flight_index": flight_index, "parent_id": parent["parent_id"],
                                 "variant": variant, "phase": phase, "phase_ordinal": ordinal, "segment_label": label,
                                 "start_relative_seconds": float(times[start]), "end_relative_seconds": float(times[end-1]),
                                 "observed_points": end-start, "sampled_points": observed,
                                 "actual_changed_points": int(point_labels[start:end].sum()),
                                 "sampled_changed_points": int(point_labels[indices].sum())})
    features, elapsed, mask = np.stack(features), np.stack(elapsed), np.stack(masks)
    flight_index, phases = np.asarray(owners, np.int64), np.asarray(phases, np.int64)
    split = np.asarray([r["split"] for r in flights])
    variants = np.asarray([r["variant"] for r in flights])
    aircraft = np.asarray([r["icao"] for r in flights])
    labels = np.asarray([r["label"] for r in flights], np.int64)
    reference_flights = (split == "train") & (variants == "clean")
    fitting_mask = reference_flights[flight_index, None] & mask
    fitting = features[fitting_mask].astype(np.float64)
    mean, scale = np.zeros(6), np.ones(6)
    mean[:5], scale[:5] = fitting[:, :5].mean(axis=0), fitting[:, :5].std(axis=0)
    scale[scale < 1e-8] = 1.
    projected = np.stack(anchors)
    lat_ref, lon_ref = float(fitting[:, 1].mean()), float(fitting[:, 0].mean())
    projected[:, :, 1] = 6371000.*np.cos(np.deg2rad(lat_ref))*np.deg2rad(projected[:, :, 1]-lon_ref)
    projected[:, :, 2] = 6371000.*np.deg2rad(projected[:, :, 2]-lat_ref)
    reference_roles = np.where(reference_flights, "train", "excluded")
    context_config = protocol["contexts"]
    peers, context = _context_peers(projected, phases, flight_index, reference_roles,
                                    context_config["candidates"], context_config["peers_per_candidate"],
                                    context_config["dbscan_eps"], context_config["dbscan_min_samples"],
                                    excluded_group_ids=aircraft[flight_index])
    valid = peers >= 0
    if not np.all(reference_flights[flight_index[peers[valid]]]):
        raise AssertionError("Non-clean or non-training context")
    features[mask] = ((features[mask].astype(np.float64)-mean)/scale).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("Invalid standardized physical features")
    output.mkdir(parents=True)
    np.savez_compressed(output/"data.npz", features=features, elapsed=elapsed, mask=mask,
                        flight_index=flight_index, labels=labels, split=split, phases=phases,
                        flight_ids=np.asarray([r["flight_id"] for r in flights]), flight_icao=aircraft, peers=peers,
                        parent_ids=np.asarray([r["parent_id"] for r in flights]), variants=variants,
                        segment_labels=np.asarray(segment_labels, np.int64))
    pd.DataFrame(flights).to_csv(output/"split.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segments.csv", index=False)
    with (output/"corruption_log.jsonl").open("x", encoding="utf-8") as handle:
        for ledger in ledgers:
            handle.write(json.dumps(ledger, ensure_ascii=False)+"\n")
    manifest = {"schema_version": 3, "dataset": scenario, "data_kind": "controlled_injection_into_supplied_raw_observations",
                "source": str(source), "source_sha256": protocol["source_sha256"],
                "protocol_path": str(protocol_path), "protocol_sha256": _sha256(protocol_path),
                "parent_split_sha256": _sha256(parent_split_path), "feature_names": FEATURE_NAMES,
                "cohort_path": str(cohort_path), "cohort_sha256": _sha256(cohort_path),
                "generation_cohort_sha256": _sha256(cohort_path),
                "generation_cohort_path": str(cohort_path),
                "implementation_addendum_sha256": _sha256(addendum_path) if addendum_path.exists() else None,
                "generation_addendum_path": str(addendum_path) if addendum_path.exists() else None,
                "generation_addendum_sha256": _sha256(addendum_path) if addendum_path.exists() else None,
                "data_sha256": _sha256(output/"data.npz"), "split_sha256": _sha256(output/"split.csv"),
                "segments_sha256": _sha256(output/"segments.csv"), "corruption_log_sha256": _sha256(output/"corruption_log.jsonl"),
                "implementation_sha256": _sha256(Path(__file__)), "shared_data_implementation_sha256": _sha256(Path(__file__).with_name("data.py")),
                "reference_policy": "training_clean_parents_only", "external_provenance_verified": False,
                "scaler": {"fit_split": "train_clean", "fit_count": len(fitting), "mean": mean.tolist(), "scale": scale.tolist(), "heading": "radians unstandardized"},
                "context": {**context, "absolute_time_origin_unix_ns": origin,
                            "projection": {"earth_radius_metres": 6371000., "reference_latitude_train_degrees": lat_ref,
                                           "reference_longitude_train_degrees": lon_ref},
                            "reference_pool": "Only train clean original parent observations, never corrupted/val/test; all target ICAO excluded.",
                            "historical_precedence_enforced": False},
                "config": {"max_points": maximum, "context_candidates": 2, "peers_per_context": 3},
                "phase_method": protocol["phase"], "point_sampling": protocol["sampling"],
                "flight_label": "0 no additional injection; 1 actual new physical corruption; original labels never read",
                "counts": {**source_counts, "retained_parents": len(parents), "retained_flights": len(flights), "retained_segments": len(features),
                           "parent_by_split": dict(Counter(p["split"] for p in parents)),
                           "flights_by_split": dict(Counter(split.tolist())), "valid_sampled_points": int(mask.sum()),
                           "sampled_changed_points": int(sum(r["sampled_changed_points"] for r in segments))},
                "limitations": protocol["limits"]+[protocol["contexts"]["paper_difference"], "Full-resolution corruptions computed then sampled; full-resolution variant CSVs are not materialized. Reproduction requires original source, protocol, seeds and logs."]}
    _write_new_json(output/"manifest.json", manifest)
    return manifest


def build_synthetic_five(source, output, scenarios=SCENARIOS):
    """Require the already frozen approved protocol; refuse replacing any output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    protocol_path = output/"generation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    actual_source_sha = _sha256(source)
    if protocol != generation_protocol(actual_source_sha):
        raise ValueError("Frozen protocol differs from approved configuration or current source")
    if not scenarios or any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Unknown or empty scenarios")
    for scenario in scenarios:
        if (output/scenario).exists():
            raise FileExistsError("Refusing overwrite %s" % (output/scenario))
    print("Reading raw physical channels (original labels excluded)", flush=True)
    frame, counts = _read_unlabelled_source(source)
    parents, excluded, origin = _parents(frame, protocol)
    del frame
    parent_records = [{key: parent[key] for key in ("parent_id", "icao", "call", "split", "start_time_utc", "end_time_utc", "start_unix_ns", "end_unix_ns", "selected_phase", "phase_seed_digest", "heading_filled_points")}
                      | {"points": len(parent["values"]), "phase_count": len(parent["ranges"])} for parent in parents]
    parent_path = output/"parent_split.csv"
    if parent_path.exists():
        saved = pd.read_csv(parent_path, dtype={"icao": str, "call": str}).fillna("")
        fresh = pd.DataFrame(parent_records).fillna("")
        if saved.to_csv(index=False) != fresh.to_csv(index=False):
            raise ValueError("Existing shared parent split differs")
    else:
        pd.DataFrame(parent_records).to_csv(parent_path, index=False, mode="x")
        _write_new_json(output/"parent_exclusions.json", excluded)
    # A pre-training technical addendum can exclude impossible support, never
    # change phase choice, roles, wave amplitudes, RNG or manufacture observations.
    unsupported = {p["parent_id"] for p in parents
                   if np.sum((p["times"] >= p["support"][0]) & (p["times"] <= p["support"][1])) < 2}
    addendum_path = output/"generation_implementation_addendum.json"
    if unsupported:
        addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
        if (addendum["generation_protocol_sha256"] != _sha256(protocol_path)
                or addendum["support_feasibility_audit_sha256"] != _sha256(output/"support_feasibility_audit.json")
                or set(addendum["excluded_parent_ids"]) != unsupported):
            raise ValueError("Technical feasibility exclusions differ from the approved frozen addendum")
    parents = [p for p in parents if p["parent_id"] not in unsupported]
    cohort_records = [r for r in parent_records if r["parent_id"] not in unsupported]
    cohort_path = output/"generation_cohort.csv"
    if cohort_path.exists():
        saved = pd.read_csv(cohort_path, dtype={"icao": str, "call": str}).fillna("")
        if saved.to_csv(index=False) != pd.DataFrame(cohort_records).fillna("").to_csv(index=False):
            raise ValueError("Existing common generation cohort differs")
    else:
        pd.DataFrame(cohort_records).to_csv(cohort_path, index=False, mode="x")
    counts.update({"prescreen_eligible_parents": len(parent_records),
                   "technical_support_exclusions": len(unsupported)})
    manifests = {}
    for scenario in scenarios:
        print("Building %s for %d paired parents" % (scenario, len(parents)), flush=True)
        manifests[scenario] = _prepare_scenario(parents, scenario, protocol, output/scenario, source, origin,
                                               counts, protocol_path, parent_path, cohort_path, addendum_path)
        print("READY %s %s" % (scenario, output/scenario), flush=True)
    return manifests
