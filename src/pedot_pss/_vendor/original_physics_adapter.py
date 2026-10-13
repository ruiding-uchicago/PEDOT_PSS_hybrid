"""Original P mechanics with clean experiments, fitted parameters and actual M.

This adapter preserves DomainTransport, its frozen-parameter C0 counterfactual,
the original two-clock GEM/interface forward, and matched arm-P's profiled
Ridge10 nuisance head with rM coefficient one. It never reads previous OOF,
source identities, audit records, saved fold states or query conductivity.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
import re

import numpy as np

from .clean_inputs import INPUT_FIELDS, canonicalize, feature_view
from . import domain_transport_v2 as domain
from . import thermal_competition_forward as forward
from . import thermal_competition_matched as matched
from . import thermal_event_ledger as ledger
from .fick_matched_screen import scale as original_common_scale

VERSION = "original-P-experiment-only-adapter-v1"
COMMON = ("log1p_n_valid_events", "log1p_total_duration_s", "min_T_K_over_403_15",
    "max_T_K_over_403_15", "n_incomplete_events", "order_assumed", "history_incomplete",
    "ambiguous_post", "M_prediction", "M_weighted_physical", "M_residual", "M_domain",
    "C0_physical", "C0_domain")
THERMAL_KINDS = frozenset({"dry", "anneal", "heat", "press"})
_LEGACY_TIMES = ("dry_duration_s", "dry_duration_min", "anneal_duration_s", "anneal_duration_min",
    "post_treatment_duration_s", "post_treatment_duration_min", "pb_first_dry_duration_s",
    "pb_first_anneal_duration_s", "pb_post_dry_duration_s", "pb_second_anneal_duration_s",
    "extra_post_dry_duration__s", "extra_second_anneal_duration__s")
_SALT = re.compile(r"\bsalt\b|\blitfsi\b|\blibr\b|\blicl\b|\blina\b|\bnacl\b|\bnabr\b|\bnai\b|"
    r"\bfebr[23]?\b|\bfecl[23]?\b|\bcubr[23]?\b|\bcocl[23]?\b|\bcacl2\b|"
    r"(?:lithium|sodium|potassium|copper|iron|cobalt|calcium)\s+(?:chloride|bromide|iodide|sulfate|sulphate|perchlorate)", re.I)
_PEO = re.compile(r"\bpeo\b|poly\s*\(?\s*ethylene\s+oxide", re.I)
_NONFILM = re.compile(r"\b(?:substrate|bare\s+glass|stock\s+solution|solvent\s+bath)\b", re.I)


def _raw(rows):
    out = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - set(INPUT_FIELDS):
            raise ValueError("Only clean public experimental fields are accepted; no IDs, outcomes or observed film data")
        out.append(canonicalize(row))
    return out


def _M(values, n):
    values = np.asarray(values)
    if values.shape != (n, 5) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Internal M must be finite n×5 model predictions, not query observations")
    if np.max(abs(values[:, 0]-values[:, 2]-values[:, 3]), initial=0.) > 1e-9:
        raise ValueError("M prediction must equal weighted physics plus residual")
    return np.asarray(values, dtype=float)


def _thermal_fact(raw):
    stages = raw.get("process_stages") or []
    if not stages:
        # Original primitive parser: unit conversion, alias conflicts, post
        # ambiguity and missing history semantics are retained exactly.
        return ledger.project(dict(raw))
    events = []
    for order, stage in enumerate(stages):
        if stage.get("kind") not in THERMAL_KINDS:
            continue
        method = str(stage.get("method") or "")
        obj = str(stage.get("object") or "")
        temp, duration = stage.get("temperature_C"), stage.get("duration_s")
        interval = bool(stage.get("temperature_range_C") or stage.get("duration_range_s"))
        path_unknown = bool(re.search(r"\bramp\b|temperature\s+sweep|(?:heating|cooling)\s+rate", method, re.I))
        nonfilm = bool(_NONFILM.search(obj)) and not re.search(r"pedot|\bfilm\b|coating|fiber|trace", obj, re.I)
        valid_T = temp is not None and math.isfinite(float(temp)) and temp > -273.15 and not interval and not path_unknown
        valid_duration = duration is not None and math.isfinite(float(duration)) and duration >= 0 and not interval and not path_unknown
        gas = ledger._atmosphere(" ".join((method, str(stage.get("atmosphere") or ""))))
        veto = (["explicit_nonfilm_thermal_event"] if nonfilm else [])
        if gas == "conflict":
            veto.append("atmosphere_conflict")
        events.append(dict(kind=stage["kind"], role=stage["kind"], order=order,
            temperature_K=float(temp)+273.15 if valid_T else None,
            duration_s=float(duration) if valid_duration else None,
            known=dict(temperature=bool(valid_T), duration=bool(valid_duration)),
            active=bool(valid_T and valid_duration and not veto), atmosphere=gas,
            vetoes=veto, pair_tier="explicit_planned_stage", interval=interval,
            path_unknown=path_unknown, conditional_scope="reported_constant_hold_only"))
    incomplete = sum(not all(event["known"].values()) for event in events)
    complete = bool(events) and incomplete == 0 and not any(event["vetoes"] for event in events)
    return dict(active=any(event["active"] for event in events), events=events,
        flags=dict(incomplete_events=incomplete, order_assumed=False,
            incomplete_processing_history=not complete, ambiguous_post_thermal_event=False),
        history_complete=complete,
        scope="Explicit planned dry/anneal/heat/press holds; ramp endpoints, ranges and missing durations are not constant holds")


def _supported_chemistry(raw, scientific):
    names = raw.get("additive_names") or []
    text = " ".join([str(raw.get("polymer_grade") or ""), str(raw.get("post_treatment") or ""),
                     *map(str, names)])
    if _PEO.search(text):
        return False, "PEO_composite_protocol_left_to_parent_M"
    if _SALT.search(text):
        return False, "salt_protocol_left_to_parent_M"
    if scientific.get("recipe_acid_present", 0.) or scientific.get("acid_post_present", 0.):
        return False, "acid_process_gate_not_generalized_without_input_based_protocol_review"
    if scientific.get("quantity_ionic", 0.) or scientific.get("quantity_mixed", 0.):
        return False, "explicit_ionic_or_mixed_transport_outside_electronic_GEM_scope"
    if scientific.get("quantity_non_pedot_reference_explicit", 0.):
        return False, "explicit_non_PEDOT_reference_left_to_parent_M"
    return True, "generic_PEDOT_PSS_GEM_interface_scope"


def _project(rows):
    raw = _raw(rows)
    scientific = [feature_view(row) for row in raw]
    facts = [_thermal_fact(row) for row in raw]
    physical = []
    chemistry, reasons = [], []
    for row, view, fact in zip(raw, scientific, facts):
        supported, reason = _supported_chemistry(row, view)
        chemistry.append(supported)
        reasons.append(reason if not supported or fact["active"] else "no_known_constant_thermal_event")
        value = dict(view)
        # Use the same actual DomainTransport coordinates, but aggregate only
        # declared unique holds. Remove every old duration alias first so the
        # original _prepare_rows cannot count those stages for a second time.
        if row.get("process_stages"):
            pairs = forward.ledger_pairs(fact)
            for key in _LEGACY_TIMES:
                value.pop(key, None)
            value.pop("anneal_temperature_C", None)
            value.pop("pb_first_anneal_temperature_C", None)
            if pairs:
                clock = forward.clock(pairs, **forward.CLOCKS["ordering"])
                # Equivalent hold at the original ordering reference; this
                # is a process coordinate, not an inferred film temperature.
                if clock["q"] is None:
                    raise ValueError("DomainTransport equivalent process hold overflow is unsupported")
                value["anneal_temperature_C"] = 130.
                value["anneal_duration_s"] = clock["q"] * forward.CLOCKS["ordering"]["tau_s"]
        physical.append(value)
    return raw, scientific, facts, physical, np.asarray(chemistry, dtype=bool), reasons


def reference_bundle(model, physical_rows):
    """Recompute original/C0 states from fitted clean parameters, without targets.

    This is the original forward.reference_states arithmetic, using the fitted
    generic DomainTransport rather than fabricating a P4 acid protocol receipt.
    No legacy acid-specific overlay, salt or PEO specialist is implicitly added.
    """
    if domain._physics.FEATURE_NAMES[6] != "thermal_exposure" or domain.PARAMETER_NAMES[6] != "thermal_barrier_response":
        raise ValueError("Original GEM/domain thermal column definitions changed")
    prepared, design, _, matrix = domain._inputs(physical_rows, model.activation_energy_eV)
    hopping, _ = domain._physics.hopping_coordinate(prepared)
    if not model.gem_.enable_hopping:
        hopping[:] = 0.
    geometry = {key: float(getattr(model.gem_, key)) for key in ("fraction", "threshold", "exponent")}
    high = np.full(len(physical_rows), model.gem_.sigma_high)
    old_log, old_low, _, _ = model.gem_._compute(design, hopping, model.gem_.coef_)
    old_parts = domain.resistor_network(old_log, matrix, model.domain_parameters_, model.mode, model.interface_scale)
    original = forward._state(old_low, high, geometry, old_parts)
    trace = dict(gem_thermal_before=design[:, 6].copy(), domain_thermal_before=matrix[:, 6].copy(),
        gem_acid_before=design[:, 3].copy(), domain_acid_before=matrix[:, 5].copy(), hopping_before=hopping.copy())
    design[:, 6] = 0.
    matrix[:, 6] = 0.
    trace.update(gem_thermal_after=design[:, 6].copy(), domain_thermal_after=matrix[:, 6].copy(),
        gem_acid_after=design[:, 3].copy(), domain_acid_after=matrix[:, 5].copy(), hopping_after=hopping.copy())
    log_sigma, low, _, _ = model.gem_._compute(design, hopping, model.gem_.coef_)
    parts = domain.resistor_network(log_sigma, matrix, model.domain_parameters_, model.mode, model.interface_scale)
    bundle = dict(version=forward.VERSION, original=original,
        C0=forward._state(low, high, geometry, parts), geometry=geometry, trace=trace)
    forward._bundle(bundle)
    return bundle


def _common(facts, M, bundle):
    parts = bundle["C0"]["reference_parts"]
    rows = []
    for i, fact in enumerate(facts):
        pairs = forward.ledger_pairs(fact)
        if not pairs:
            raise ValueError("COMMON14 is defined on admitted known thermal histories only")
        flags = fact["flags"]
        temperatures = [event["temperature_K"] for event in pairs]
        rows.append([math.log1p(len(pairs)), math.log1p(math.fsum(event["duration_s"] for event in pairs)),
            min(temperatures)/403.15, max(temperatures)/403.15,
            float(flags["incomplete_events"]), float(flags["order_assumed"]),
            float(flags["incomplete_processing_history"]), float(flags["ambiguous_post_thermal_event"]),
            *M[i, [0, 2, 3, 4]], parts["prediction_log10_S_cm"][i], parts["domain_correction_log10"][i]])
    return np.asarray(rows, dtype=float).reshape(len(facts), 14)


def _coordinates(facts):
    return forward.clock_coordinates([forward.ledger_pairs(fact) for fact in facts],
                                     admitted=np.ones(len(facts), dtype=bool))


def fit_original_physics(raw_train, M_train, y_train, *, fold=0, max_nfev=150):
    """Fit the original generic GEM/circuit and arm-P only on clean TRAIN data.

    Returns complete state, TRAIN n×5 predictions and applicability diagnostics.
    A branch with inadequate two-clock support remains explicitly unavailable
    and copies M; it does not fabricate clock observations or fitted parameters.
    """
    raw, scientific, facts, physical, chemistry, reasons = _project(raw_train)
    M = _M(M_train, len(raw))
    y = np.asarray(y_train)
    if y.shape != (len(raw),) or y.dtype.kind not in "iuf" or not np.isfinite(y).all() or not len(raw):
        raise ValueError("Finite nonempty TRAIN-only conductivity targets required")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold is an internal RNG index in 0..4")
    state = dict(version=VERSION, domain_model=None, thermal_fit=None, common_scaler=None,
        common_definition=COMMON, domain_parameter_summary=None,
        specialized_overlays="acid/salt/PEO protocol-specific laws not generalized; unsupported rows retain parent M",
        clock_scope="Original two cumulative Arrhenius severity clocks; damage clock is not an identified air-only oxidation law",
        rM_coefficient=1., n_train=len(raw), n_domain_train=int(chemistry.sum()))
    if not chemistry.any():
        state["status"] = "no_generic_domain_TRAIN_support"
        return state, M.copy(), dict(eligible=np.zeros(len(raw), dtype=bool), reasons=reasons)
    model = domain.DomainTransport(mode="anisotropic_bypass", coupling="joint", max_nfev=max_nfev,
        random_state=20261009+fold, regularization=.03, gem_regularization=.01,
        activation_energy_eV=.35, interface_scale=1.)
    model.fit([row for row, use in zip(physical, chemistry) if use], y[chemistry])
    state["domain_model"] = model
    state["domain_parameter_summary"] = model.summary()
    mask = chemistry & np.asarray([fact["active"] for fact in facts], dtype=bool)
    if mask.sum() < matched.MIN_TRAIN:
        state["status"] = "insufficient_admitted_TRAIN_for_original_P"
        return state, M.copy(), dict(eligible=mask, reasons=reasons)
    index = np.flatnonzero(mask)
    selected_facts = [facts[i] for i in index]
    bundle = reference_bundle(model, [physical[i] for i in index])
    coordinates = _coordinates(selected_facts)
    H = np.column_stack((coordinates["ordering_H"], coordinates["damage_H"]))
    singular = np.linalg.svd(H-H.mean(0), compute_uv=False)
    rank = int(np.sum(singular > 1e-10 * max(1., float(singular[0]))))
    state["clock_rank"] = rank
    if rank != 2:
        state["status"] = "two_independent_original_clock_directions_unavailable"
        return state, M.copy(), dict(eligible=mask, reasons=reasons)
    common = _common(selected_facts, M[index], bundle)
    X, _, scaler = original_common_scale(common, common[:0])
    fit = matched.fit_arm(bundle, coordinates, X, M[index, 3], y[index], "P")
    state.update(status="original_P_TRAIN_fitted", thermal_fit=fit, common_scaler=scaler)
    values = M.copy()
    values[index], parts = matched.predict_arm(bundle, coordinates, X, M[index, 3], fit)
    checks = dict(gem_equation_max_abs=float(np.max(abs(parts["gem_equation_residual"]), initial=0.)),
        circuit_closure_max_abs=float(np.max(abs(parts["closure_residual"]), initial=0.)),
        n_bound_violations=int(parts["domain_bound_violation"].sum()+parts["gem_phase_bound_violation"].sum()))
    return state, values, dict(eligible=mask, reasons=reasons, physics_checks=checks)


def predict_original_physics(state, raw_query, M_query):
    """QUERY recipe plus internal M → n×5; no query y or source-specific routing."""
    if not isinstance(state, dict) or state.get("version") != VERSION:
        raise ValueError("Complete original P adapter state required")
    raw, scientific, facts, physical, chemistry, reasons = _project(raw_query)
    M = _M(M_query, len(raw))
    values = M.copy()
    eligible = chemistry & np.asarray([fact["active"] for fact in facts], dtype=bool)
    if state.get("thermal_fit") is None or not eligible.any():
        return values, dict(eligible=np.zeros(len(raw), dtype=bool), reasons=reasons,
                            status=state["status"])
    index = np.flatnonzero(eligible)
    selected_facts = [facts[i] for i in index]
    bundle = reference_bundle(state["domain_model"], [physical[i] for i in index])
    coordinates = _coordinates(selected_facts)
    common = _common(selected_facts, M[index], bundle)
    scaler = state["common_scaler"]
    X = (common-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])
    values[index], parts = matched.predict_arm(bundle, coordinates, X, M[index, 3], state["thermal_fit"])
    return values, dict(eligible=eligible, reasons=reasons,
        physical_parts=parts, C0=bundle["C0"], coordinates=coordinates,
        scope="Reported known constant holds; omitted history is flagged, not imputed to zero")
