"""Read-only replay and coordinate maps for column-permutation study v1.

This module performs no fitting and does not change the frozen generators.
All new targets are exact replays of previously saved X/y/truth/lambda; the
historical coefficient vectors remain diagnostic outputs, never warm starts.
"""
from __future__ import annotations

import copy
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from .data import array_sha256, file_sha256
from .data_v2 import make_uji
from .workshop_data import make_real_design
from .audit_data_v1 import load_audit_source, make_audit_design


ROOT = Path(__file__).resolve().parents[2]


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def _display(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normal_params(params):
    source = params["source"]
    if source not in {"uji", "news", "gene"}:
        raise ValueError("source must be uji, news or gene.")
    seed = _integer(params["seed"], "seed")
    amplitude = params["amplitude"]
    if amplitude not in {"strong", "mixed_weak"}:
        raise ValueError("amplitude must be strong or mixed_weak.")
    layout = params.get("layout")
    if isinstance(params.get("c", 1.), (bool, np.bool_)):
        raise ValueError("c must be a numeric regularization factor, not a boolean.")
    c = float(params.get("c", 1.))
    if source == "uji":
        if layout not in {"dispersed", "clustered"} or c != 1.:
            raise ValueError("UJI replay retains its old layout and fixed c=1.")
    elif layout is not None or c not in {.5, 1.}:
        raise ValueError("External replay has layout=None and c in {0.5, 1}.")
    return {"source": source, "seed": seed, "layout": layout, "amplitude": amplitude, "c": c}


def target_parameters(config):
    """Return the registered 40 UJI + 8 news + 8 gene paired targets."""
    sources = config.get("sources", ["uji", "news", "gene"])
    if len(sources) != 3 or set(sources) != {"uji", "news", "gene"}:
        raise ValueError("The study retains all three registered sources.")
    amplitudes = config.get("amplitudes", ["strong", "mixed_weak"])
    layouts = config.get("layouts", ["dispersed", "clustered"])
    if set(amplitudes) != {"strong", "mixed_weak"} or len(amplitudes) != 2 or set(layouts) != {"dispersed", "clustered"} or len(layouts) != 2:
        raise ValueError("Both old amplitude/layout conditions must be retained.")
    if list(config.get("external_c", [.5, 1.])) != [.5, 1.]:
        raise ValueError("The registered external c levels are [0.5, 1.0].")
    seeds = {"uji": config.get("uji_seeds", list(range(300, 310))),
             "news": config.get("external_seeds", {}).get("news", [401, 402]),
             "gene": config.get("external_seeds", {}).get("gene", [501, 502])}
    registered = {"uji": list(range(300, 310)), "news": [401, 402], "gene": [501, 502]}
    if any(list(seeds[source]) != registered[source] for source in registered):
        raise ValueError("The frozen target seeds are UJI300–309, news401/402 and gene501/502.")
    output = []
    for source in sources:
        source_layouts = layouts if source == "uji" else [None]
        c_values = [1.] if source == "uji" else [.5, 1.]
        for seed, layout, amplitude, c in product(seeds[source], source_layouts, amplitudes, c_values):
            output.append(_normal_params(dict(source=source, seed=seed, layout=layout, amplitude=amplitude, c=c)))
    if len(output) != config.get("expected_targets", 56) or len({target_id(p) for p in output}) != len(output):
        raise ValueError("Target identifiers/grid differ from the registered 56 targets.")
    return output


def target_id(params):
    """Source-prefixed ID, matching the original saved target basename."""
    p = _normal_params(params)
    if p["source"] == "uji":
        return f"uji_seed{p['seed']}_{p['layout']}_{p['amplitude']}"
    return f"{p['source']}_seed{p['seed']}_c{p['c']:g}_{p['amplitude']}"


def load_sources(config):
    """Prepare each frozen source once; caller owns any measured setup timer."""
    uji_config_path = _path(config.get("uji_config_path", "configs/workshop_real.json"))
    external_config_path = _path(config.get("external_config_path", "configs/reference_audit_v1.json"))
    paths = config.get("baseline_paths", {})
    uji_config, external_config = _read(uji_config_path), _read(external_config_path)
    prepared = {"uji": make_uji(seed=0), "news": load_audit_source("news", source_seed=0),
                "gene": load_audit_source("gene", source_seed=0)}
    output = {}
    for source in ("uji", "news", "gene"):
        path = uji_config_path if source == "uji" else external_config_path
        output[source] = {"source": source, "prepared": prepared[source],
                          "baseline_config": uji_config if source == "uji" else external_config,
                          "baseline_config_path": path, "baseline_config_sha256": file_sha256(path),
                          "baseline_path": _path(paths.get(source, "results/workshop/real" if source == "uji" else "results/reference_audit_v1/confirm"))}
    return output


def make_target(config, params, sources):
    """Regenerate one old problem and validate its saved data/reference linkage.

    Return X/y/w_true/metadata plus historical_reference and origin dictionaries.
    Historical coefficients are in original coordinates and must not initialize
    any new fit. UJI numerical-reference uncertainty is preserved.
    """
    params = _normal_params(params); source = params["source"]; identifier = target_id(params)
    context = sources[source]
    if context.get("source", source) != source:
        raise ValueError("Source context does not match requested target.")
    base = Path(context["baseline_path"]).resolve()
    dataset_path, reference_path = base/"datasets"/(identifier+".json"), base/"references"/(identifier+".json")
    old_ds, old_ref = _read(dataset_path), _read(reference_path)
    baseline_config_path = Path(context["baseline_config_path"]).resolve()
    current_config_hash = file_sha256(baseline_config_path)
    if (current_config_hash != context["baseline_config_sha256"]
            or current_config_hash != old_ds["config_sha256"]
            or current_config_hash != old_ref["config_sha256"]
            or _read(baseline_config_path) != context["baseline_config"]):
        raise ValueError("Current historical config, loaded context and frozen dataset/reference config hashes differ.")
    coefficient_relative = Path(old_ref.get("coefficient_path", f"reference_coefficients/{identifier}.npz"))
    coefficient_path = (base/coefficient_relative).resolve()
    if coefficient_relative.is_absolute() or not coefficient_path.is_relative_to(base) or coefficient_path != base/"reference_coefficients"/(identifier+".npz"):
        raise ValueError("Historical reference coefficient path escapes its registered target.")
    if file_sha256(coefficient_path) != old_ref["coefficient_sha256"]:
        raise ValueError("Historical coefficient file differs from the saved reference hash.")
    old_config = context["baseline_config"]
    if source == "uji":
        generated = make_real_design(old_config, params["seed"], params["layout"], params["amplitude"], source=context["prepared"])
    else:
        generated = make_audit_design(old_config, params["seed"], source, params["amplitude"], params["c"], source_data=context["prepared"])
    metadata = generated["metadata"]
    if metadata["hashes"] != old_ds["hashes"]:
        raise ValueError("Regenerated target hashes differ from the frozen dataset.")
    if metadata["lambda"] != old_ds["lambda"] or old_ref["lambda"] != old_ds["lambda"]:
        raise ValueError("Regenerated, historical dataset and historical reference lambda differ.")
    data_hash = hashlib.sha256(json.dumps(metadata["hashes"], sort_keys=True).encode()).hexdigest()
    if old_ref["dataset_hash"] != data_hash or old_ds.get("dataset_hash", data_hash) != data_hash:
        raise ValueError("Historical dataset/reference hash linkage differs.")
    if old_ref["numerical_signature"] != old_ds["numerical_signature"] or old_ref["config_sha256"] != old_ds["config_sha256"]:
        raise ValueError("Historical dataset/reference source or config differs.")
    for key in ("seed", "amplitude"):
        if old_ds[key] != params[key]:
            raise ValueError("Historical target identity differs.")
    if source == "uji" and old_ds["layout"] != params["layout"]:
        raise ValueError("Historical UJI layout differs.")
    if source != "uji" and (old_ds["source"] != source or old_ds["c"] != params["c"]):
        raise ValueError("Historical external source/c differs.")
    with np.load(coefficient_path, allow_pickle=False) as arrays:
        historical = {key: arrays[key].copy() for key in ("ref_sklearn", "ref_skglm")}
        if not np.array_equal(arrays["w_true"], generated["w_true"]):
            raise ValueError("Historical and regenerated true coefficients differ.")
    p = generated["X"].shape[1]
    if any(values.dtype != np.float64 or values.shape != (p,) or not np.isfinite(values).all() for values in historical.values()):
        raise ValueError("Historical reference coefficients have invalid shape, dtype or values.")
    if any(array_sha256(generated[name]) != metadata["hashes"][hash_key]
           for name, hash_key in (("X", "X"), ("y", "y"), ("w_true", "w_true"))):
        raise ValueError("Generated arrays differ from their recorded hashes.")
    existing_grouping = base/"grouping"/(metadata["hashes"]["X"]+".json") if source != "uji" else None
    origin = {"old_target_id": identifier, "baseline_path": _display(base),
              "dataset_path": _display(dataset_path), "dataset_sha256": file_sha256(dataset_path),
              "reference_path": _display(reference_path), "reference_sha256": file_sha256(reference_path),
              "coefficient_path": _display(coefficient_path), "coefficient_sha256": file_sha256(coefficient_path),
              "historical_reference_accepted": bool(old_ref["reference_accepted"]),
              "historical_numerical_signature": old_ref["numerical_signature"], "historical_config_sha256": old_ref["config_sha256"],
              "baseline_config_path": _display(baseline_config_path), "baseline_config_sha256": current_config_hash,
              "existing_grouping_path": _display(existing_grouping) if existing_grouping is not None and existing_grouping.is_file() else None,
              "existing_uji_grouping_metadata_path": _display(base/"reference_diagnostics.json") if source == "uji" and (base/"reference_diagnostics.json").is_file() else None}
    replay_meta = copy.deepcopy(metadata)
    replay_meta.update(source=source, seed=params["seed"], layout=params["layout"], amplitude=params["amplitude"], c=params["c"],
                       target_id=identifier, historical_dataset_hash=data_hash, original_reference_accepted=bool(old_ref["reference_accepted"]),
                       coefficient_coordinates="Original frozen final-X coordinates; no new preprocessing or grouping")
    return {"X": generated["X"], "y": generated["y"], "w_true": generated["w_true"], "metadata": replay_meta,
            "historical_reference": {"metadata": old_ref, **historical, "path": origin["reference_path"],
                                     "coefficient_path": origin["coefficient_path"], "coefficient_sha256": origin["coefficient_sha256"]},
            "origin": origin}


def _permutation(permutation, p):
    values = np.asarray(permutation)
    if values.ndim != 1 or values.shape != (p,) or not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        raise ValueError("permutation must be a one-dimensional integer array of length p.")
    values = values.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(values), np.arange(p, dtype=np.int64)):
        raise ValueError("permutation must contain each original column index exactly once.")
    return values


def make_permutations(p, seeds=(700, 701, 702, 703, 704), *, replacement_seed_start=705):
    """Identity plus five distinct fixed random orders, with deterministic replacements.

    JSON-safe records carry both arrays and their shape/dtype/value hashes. The
    extra original-order QC repetition is deliberately not one of these orders.
    """
    p = _integer(p, "p", 1)
    seeds = tuple(_integer(seed, "permutation seed") for seed in seeds)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct requested permutation seeds are required.")
    if p < 3:
        raise ValueError("At least three columns are needed for six distinct orders.")
    next_seed = _integer(replacement_seed_start, "replacement_seed_start", 705)
    if next_seed <= max(seeds):
        raise ValueError("Replacement seed range must follow all requested seeds.")
    identity = np.arange(p, dtype=np.int64); seen = {tuple(identity.tolist())}
    def record(order_id, values, requested, actual):
        inverse = np.empty(p, dtype=np.int64); inverse[values] = identity
        return {"order_id": order_id, "permutation": values.tolist(), "inverse_permutation": inverse.tolist(),
                "requested_seed": requested, "seed": actual, "replacement_used": requested != actual,
                "permutation_sha256": array_sha256(values), "inverse_sha256": array_sha256(inverse)}
    result = [record("identity", identity, None, None)]
    for index, requested in enumerate(seeds):
        actual = requested
        while True:
            candidate = np.asarray(np.random.default_rng(actual).permutation(p), dtype=np.int64)
            key = tuple(candidate.tolist())
            if key not in seen:
                break
            actual = next_seed; next_seed += 1
        seen.add(key); result.append(record(f"perm_{index}", candidate, requested, actual))
    return result


def inverse_coefficients(coef_permuted, permutation):
    """Restore original columns by scatter: w_original[..., perm]=w_permuted."""
    coefficients = np.asarray(coef_permuted, dtype=np.float64)
    if coefficients.ndim < 1 or not np.isfinite(coefficients).all():
        raise ValueError("Coefficients need a finite trailing feature dimension.")
    columns = _permutation(permutation, coefficients.shape[-1])
    original = np.empty_like(coefficients)
    original[..., columns] = coefficients
    return original


def permuted_data(data, permutation):
    """Reorder final X/truth, copying unchanged y and retaining the same lambda."""
    X, y, truth = (np.asarray(data[name], dtype=np.float64) for name in ("X", "y", "w_true"))
    if X.ndim != 2 or y.shape != (len(X),) or truth.shape != (X.shape[1],) or not all(np.isfinite(a).all() for a in (X, y, truth)):
        raise ValueError("Invalid original X/y/w_true arrays.")
    columns = _permutation(permutation, X.shape[1])
    permuted_X = np.asfortranarray(X[:, columns]); permuted_truth = np.ascontiguousarray(truth[columns]); copied_y = y.copy()
    hashes = {"X": array_sha256(permuted_X), "y": array_sha256(copied_y), "w_true": array_sha256(permuted_truth)}
    return {"X": permuted_X, "y": copied_y, "w_true": permuted_truth,
            "metadata": {"lambda": data["metadata"]["lambda"], "hashes": hashes,
                         "original_input_hashes": {key: array_sha256(value) for key, value in (("X", X), ("y", y), ("w_true", truth))},
                         "permutation_sha256": array_sha256(columns),
                         "coefficient_coordinates": "Permuted final-X columns; inverse scatter restores original variable identities",
                         "preprocessing_repeated": False, "response_changed": False, "lambda_changed": False}}
