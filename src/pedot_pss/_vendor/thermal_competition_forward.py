"""Two-clock effective-phase/interface hypothesis; pure SOURCE, no fit or I/O.

C0 removes both old thermal columns with frozen SAME-fold P4 parameters; it is
not a refitted no-thermal model. Equal phase scaling preserves contrast and GEM
homogeneity, not a new percolation geometry. Damage is bounded finite-window
state scaling, NOT exp(-kd*t) lifetime. Ordering kinetics and transfer of the
air-aging damage scenario to other protocols are explicit reference hypotheses.
No inversion of a measured/weighted/ML conductivity, identified kinetics or ODE.
"""
from collections.abc import Mapping
import math
from numbers import Real
import sys

import numpy as np

from . import correlated_domain_reference as reference
from . import solvent_drying_boundary as boundary
VERSION = "two-clock-frozen-P4-GEM-interface-v1"
KB_EV_K = 8.617333262e-5
CLOCKS = {"ordering": dict(energy_eV=.35, tau_s=600., reference_temperature_K=403.15),
          "damage": dict(energy_eV=.4, tau_s=1./2.78e-5, reference_temperature_K=393.15)}
CORE_KEYS = frozenset(("prediction_log10_S_cm", "gem_log10", "domain_correction_log10",
    "series_correction_log10", "sigma_domain_S_cm", "sigma_gem_S_cm", "sigma_series_S_cm",
    "series_resistivity_ohm_cm", "barrier_rho_ohm_cm", "anisotropy_factor", "bypass_fraction",
    "closure_residual", "domain_bound_violation", "gem_equation_residual", "gem_phase_bound_violation"))
CHECK_TOLERANCE = 1e-10


def configuration():
    return dict(version=VERSION, clocks={k: dict(v) for k, v in CLOCKS.items()},
        Boltzmann_eV_K=KB_EV_K, amplitude_bounds={"A": [0., 4.], "B": [0., 4.]},
        zero_based_thermal_columns=dict(GEM=6, domain=6),
        scope=__doc__.strip(), unsupported="COPY original P4, never C0",
        clock_scope="permutation-invariant cumulative reference clocks, not sequence ODE")


def _number(value, name, *, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Nonboolean number required: " + name)
    try: out = float(value)
    except (ValueError, TypeError, OverflowError) as exc: raise ValueError(name) from exc
    if not math.isfinite(out) or (lower is not None and out < lower) or (upper is not None and out > upper):
        raise ValueError("Finite bounded number required: " + name)
    return out


def _array(value, n, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (n,): raise ValueError("Aligned vector required: " + name)
    return np.asarray([_number(v, name) for v in raw], float)


def _mask(value, n):
    raw = np.asarray(value)
    if raw.shape != (n,) or raw.dtype != bool: raise ValueError("Aligned Boolean admission required")
    return raw.copy()


def clock(events, *, energy_eV, tau_s, reference_temperature_K):
    """Nonempty, known typed {temperature_K,duration_s} pairs; no alias inference.

    Empty/unknown is NOT zero. Explicit zero durations give q=H=0/log_q=None.
    Positive underflow fails closed. Unrepresentably large q retains finite
    log_q, q=None and q_overflow=True; its representable saturation H is one.
    """
    energy = _number(energy_eV, "energy_eV", lower=0.)
    tau = _number(tau_s, "tau_s"); tref = _number(reference_temperature_K, "Tref_K")
    if tau <= 0. or tref <= 0.: raise ValueError("Positive tau/reference K required")
    if type(events) is not list or not events: raise ValueError("Unknown/empty events are not zero exposure")
    logs = []
    for event in events:
        if type(event) is not dict or set(event) != {"temperature_K", "duration_s"}:
            raise ValueError("Exactly two typed event quantities required")
        T = _number(event["temperature_K"], "temperature_K")
        seconds = _number(event["duration_s"], "duration_s", lower=0.)
        if T <= 0.: raise ValueError("Strictly positive absolute temperature required")
        if seconds == 0.: continue
        shift = energy/KB_EV_K*(1./T-1./tref) if energy else 0.
        logq = math.log(seconds)-math.log(tau)-shift
        if not math.isfinite(logq): raise ValueError("Unrepresentable event log exposure")
        logs.append(logq)
    if not logs:
        return dict(q=0., H=0., log_q=None, q_overflow=False, n_events=len(events), n_positive_duration=0)
    peak = max(logs); logq = peak+math.log(math.fsum(math.exp(v-peak) for v in logs))
    if not math.isfinite(logq): raise ValueError("Unrepresentable summed log exposure")
    overflow = logq > math.log(sys.float_info.max)
    q = None if overflow else math.exp(logq)
    if q == 0.: raise ValueError("Positive exposure underflows representable q/H; not unknown or zero")
    H = 1. if overflow else -math.expm1(-q)
    if not math.isfinite(H) or H <= 0. or H > 1.: raise ValueError("Positive finite exposure saturation required")
    return dict(q=q, H=H, log_q=logq, q_overflow=overflow,
                n_events=len(events), n_positive_duration=len(logs))


def clock_coordinates(events_by_row, *, admitted):
    """Caller supplies ledger-admitted unique active pairs; inactive stays None.

    Ledger metadata/unknown atmosphere remain caller-owned, not asserted air.
    Incomplete extra stages must not be converted into zero-duration events.
    """
    if type(events_by_row) is not list: raise ValueError("Aligned event rows required")
    mask = _mask(admitted, len(events_by_row)); order, damage, audit = [], [], []
    for events, active in zip(events_by_row, mask):
        if not active:
            order.append(None); damage.append(None); audit.append(None); continue
        pair = {k: clock(events, **cfg) for k, cfg in CLOCKS.items()}
        order.append(pair["ordering"]["H"]); damage.append(pair["damage"]["H"]); audit.append(pair)
    return dict(version=VERSION, admitted=mask, ordering_H=order, damage_H=damage, clock_audit=audit)


def ledger_pairs(fact):
    """Read ledger's active/known contract only; never re-infer native admission.

    [] means unsupported/unknown, NOT zero exposure. The caller must retain the
    ledger's atmosphere/history metadata and pass its admission mask separately.
    Explicit duration zero is kept; incomplete additional events stay unknown.
    """
    if not isinstance(fact, Mapping) or type(fact.get("active")) is not bool or type(fact.get("events")) is not list:
        raise ValueError("Typed ledger facts required")
    if not fact["active"]: return []
    pairs = []
    for event in fact["events"]:
        if type(event) is not dict or type(event.get("active")) is not bool: raise ValueError("Typed event active flag required")
        if not event["active"]: continue
        known = event.get("known")
        if (type(known) is not dict or set(known) != {"temperature", "duration"} or
                any(type(v) is not bool or not v for v in known.values())):
            raise ValueError("Active event must have two known quantities")
        if event.get("vetoes") != [] or event.get("atmosphere") == "conflict": raise ValueError("Conflicting active event")
        T = _number(event.get("temperature_K"), "temperature_K")
        duration = _number(event.get("duration_s"), "duration_s", lower=0.)
        if T <= 0.: raise ValueError("Strictly positive event K required")
        pairs.append(dict(temperature_K=T, duration_s=duration))
    if not pairs: raise ValueError("Admitted ledger row has no known active event")
    return pairs


def _coordinates(value, n):
    if type(value) is not dict or set(value) != {"version", "admitted", "ordering_H", "damage_H", "clock_audit"} or value["version"] != VERSION:
        raise ValueError("Closed two-clock coordinates required")
    mask = _mask(value["admitted"], n); out = []
    if type(value["clock_audit"]) is not list or len(value["clock_audit"]) != n: raise ValueError("Aligned clock audit required")
    for name in ("ordering", "damage"):
        raw = value[name+"_H"]
        if type(raw) is not list or len(raw) != n: raise ValueError("Aligned nullable H required")
        a = np.zeros(n)
        for i, active in enumerate(mask):
            if not active:
                if raw[i] is not None or value["clock_audit"][i] is not None: raise ValueError("Unknown clock must remain None")
                continue
            a[i] = _number(raw[i], name+" H", lower=0., upper=1.)
            pair = value["clock_audit"][i]
            if (type(pair) is not dict or set(pair) != set(CLOCKS) or type(pair[name]) is not dict or
                    _number(pair[name].get("H"), name+" audit H", lower=0., upper=1.) != a[i]):
                raise ValueError("Clock audit/H disagree")
        out.append(a)
    return mask, *out


def _certify(parts, low, high, geometry):
    n = len(low); out = {k: _array(parts[k], n, k) for k in CORE_KEYS-{"gem_equation_residual", "gem_phase_bound_violation"}}
    if np.any(low <= 0.) or np.any(high < low): raise ValueError("Positive ordered phases required")
    g, sigma, series = (out[k] for k in ("sigma_gem_S_cm", "sigma_domain_S_cm", "sigma_series_S_cm"))
    rho, bypass = out["series_resistivity_ohm_cm"], out["bypass_fraction"]
    if (np.any(g <= 0.) or np.any(sigma <= 0.) or np.any(series <= 0.) or np.any(rho < 0.) or
            np.any((bypass < 0.) | (bypass > 1.)) or np.any(out["anisotropy_factor"] <= 0.)):
        raise ValueError("Positive phase/circuit quantities required")
    residual = reference.domain._physics.gem_residual(g, low, high, **geometry)
    phase = ((g < low*(1.-1e-11)) | (g > high*(1.+1e-11))).astype(float)
    expected_series = g/(1.+rho*g); expected = bypass*g+(1.-bypass)*expected_series
    if (not np.isfinite(residual).all() or np.max(abs(residual), initial=0.) > CHECK_TOLERANCE or phase.any() or
            out["domain_bound_violation"].any() or np.max(abs(out["closure_residual"]), initial=0.) > CHECK_TOLERANCE or
            not np.array_equal(rho, out["barrier_rho_ohm_cm"]) or
            not np.allclose(series, expected_series, rtol=CHECK_TOLERANCE, atol=0.) or
            not np.allclose(sigma, expected, rtol=CHECK_TOLERANCE, atol=0.) or
            not np.allclose(out["series_correction_log10"], -np.log1p(rho*g)/math.log(10.), rtol=0., atol=CHECK_TOLERANCE) or
            not np.allclose(out["gem_log10"], np.log10(g), rtol=0., atol=CHECK_TOLERANCE) or
            not np.allclose(out["prediction_log10_S_cm"], np.log10(sigma), rtol=0., atol=CHECK_TOLERANCE) or
            not np.allclose(out["prediction_log10_S_cm"], out["gem_log10"]+out["domain_correction_log10"], rtol=0., atol=CHECK_TOLERANCE)):
        raise ValueError("Actual GEM/circuit equation, aliases or bounds failed")
    if ("gem_equation_residual" in parts and not np.allclose(_array(parts["gem_equation_residual"], n, "GEM residual"), residual, rtol=0., atol=1e-12) or
            "gem_phase_bound_violation" in parts and not np.array_equal(_array(parts["gem_phase_bound_violation"], n, "phase bounds"), phase)):
        raise ValueError("Supplied GEM certificates differ from actual state")
    out.update(gem_equation_residual=residual, gem_phase_bound_violation=phase)
    return out


def _state(low, high, geometry, parts):
    core = _certify(parts, low, high, geometry); n = len(low)
    return dict(effective_low_phase_S_cm=low.copy(), high_sigma_S_cm=high.copy(),
        occupancy=np.full(n, geometry["fraction"]), contrast=low/high,
        rho_ohm_cm=core["series_resistivity_ohm_cm"].copy(), p_bypass=core["bypass_fraction"].copy(),
        reference_parts=core)


def reference_states(physical_rows, physics_summary, gate_summary, acid_rows, protocol_admitted):
    """Actual original P4 and separately recalculated, frozen-parameter C0.

    Only small SAME-fold parameter summaries, never records/values/ML teacher.
    Acid overlay and hopping are unchanged BEFORE removing both thermal columns.
    """
    physical, acids = list(physical_rows), list(acid_rows)
    original = reference.reference_state(physical, physics_summary, gate_summary, acids, protocol_admitted)
    base = reference._base(physics_summary, gate_summary); domain = reference.domain
    if domain._physics.FEATURE_NAMES[6] != "thermal_exposure" or domain.PARAMETER_NAMES[6] != "thermal_barrier_response":
        raise ValueError("Frozen thermal column schema changed")
    prepared, design, _, matrix = domain._inputs(physical, base.activation_energy_eV)
    acid = original["reference_parts"]["acid_process_new_coordinate"]
    design[:, reference.gate.GEM_ACID_COLUMN] = acid
    matrix[:, reference.gate.DOMAIN_ACID_COLUMN] = -acid
    hopping, _ = domain._physics.hopping_coordinate(prepared)
    if not base.gem_.enable_hopping: hopping[:] = 0.
    trace = dict(gem_thermal_before=design[:, 6].copy(), domain_thermal_before=matrix[:, 6].copy(),
        gem_acid_before=design[:, 3].copy(), domain_acid_before=matrix[:, 5].copy(), hopping_before=hopping.copy())
    design[:, 6] = 0.; matrix[:, 6] = 0.
    trace.update(gem_thermal_after=design[:, 6].copy(), domain_thermal_after=matrix[:, 6].copy(),
        gem_acid_after=design[:, 3].copy(), domain_acid_after=matrix[:, 5].copy(), hopping_after=hopping.copy())
    glog, low, _, _ = base.gem_._compute(design, hopping, base.gem_.coef_)
    geometry = {k: float(getattr(base.gem_, k)) for k in ("fraction", "threshold", "exponent")}
    high = np.full(len(physical), base.gem_.sigma_high)
    parts = domain.resistor_network(glog, matrix, base.domain_parameters_, base.mode, base.interface_scale)
    C0 = _state(low, high, geometry, parts)
    return dict(version=VERSION, original=original, C0=C0, geometry=geometry, trace=trace)


def _bundle(bundle):
    if type(bundle) is not dict or set(bundle) != {"version", "original", "C0", "geometry", "trace"} or bundle["version"] != VERSION:
        raise ValueError("Closed original/C0 reference bundle required")
    geometry = bundle["geometry"]
    if type(geometry) is not dict or set(geometry) != {"fraction", "threshold", "exponent"}: raise ValueError("Exact GEM geometry required")
    f, fc, t = [_number(geometry[k], k) for k in ("fraction", "threshold", "exponent")]
    if not 0. <= f <= 1. or not 0. < fc < 1. or t <= 0.: raise ValueError("Invalid GEM geometry")
    n = len(bundle["C0"]["effective_low_phase_S_cm"]); states = []
    for name in ("original", "C0"):
        state = bundle[name]
        low, high, rho, bypass = [_array(state[k], n, k) for k in
            ("effective_low_phase_S_cm", "high_sigma_S_cm", "rho_ohm_cm", "p_bypass")]
        if (not np.array_equal(_array(state["occupancy"], n, "occupancy"), np.full(n, f)) or
                not np.array_equal(_array(state["contrast"], n, "contrast"), low/high)):
            raise ValueError("Reference fraction/contrast differs")
        core = _certify(state["reference_parts"], low, high, geometry)
        if not np.array_equal(rho, core["series_resistivity_ohm_cm"]) or not np.array_equal(bypass, core["bypass_fraction"]):
            raise ValueError("Reference effective circuit aliases differ")
        # Preserve the original certified core byte-for-byte, including nulls.
        states.append((low, high, {k: _array(state["reference_parts"][k], n, k) for k in CORE_KEYS}))
    trace = bundle["trace"]
    for prefix in ("gem_thermal", "domain_thermal", "gem_acid", "domain_acid", "hopping"):
        before, after = [_array(trace[prefix+"_"+side], n, prefix) for side in ("before", "after")]
        changed = bool(after.any()) if "thermal" in prefix else not np.array_equal(before, after)
        if changed:
            raise ValueError("C0 must remove BOTH thermal columns and preserve acid/hopping")
    return n, states, geometry


def apply_competition(bundle, coordinates, *, A, B):
    """A/B in[0,4]; admitted null COPY C0, unsupported COPY original P4.

    New phase/rho state enters actual GEM and effective-rho network ONCE.
    Delta versus C0 on unsupported rows is only a reference difference, not
    an applied correction; delta versus original there is exactly zero.
    """
    A = _number(A, "A", lower=0., upper=4.); B = _number(B, "B", lower=0., upper=4.)
    n, states, geometry = _bundle(bundle); mask, Ho, Hd = _coordinates(coordinates, n)
    old_low, old_high, old = states[0]; low0, high0, C0 = states[1]
    result = {k: v.copy() for k, v in old.items()}
    for k in result: result[k][mask] = C0[k][mask]
    low, high, rho = old_low.copy(), old_high.copy(), old["series_resistivity_ohm_cm"].copy()
    scale = np.exp(-B*Hd[mask]); relief = np.exp(-A*Ho[mask])
    low[mask], high[mask] = low0[mask]*scale, high0[mask]*scale
    rho[mask] = C0["series_resistivity_ohm_cm"][mask]*relief
    if np.any(low <= 0.) or np.any(high < low) or np.any((C0["series_resistivity_ohm_cm"] > 0.) & mask & (rho <= 0.)):
        raise ValueError("Modified positive phase/rho state not representable")
    changed = mask & ((low != low0) | (high != high0) | (rho != C0["series_resistivity_ohm_cm"]))
    homogeneity_error = 0.
    if changed.any():
        g = reference.domain._physics.gem_conductivity(low[changed], high[changed], **geometry)
        parts = boundary.network_with_rho(np.log10(g), rho[changed], C0["bypass_fraction"][changed],
            anisotropy_factor=C0["anisotropy_factor"][changed])
        corrected = _certify(parts, low[changed], high[changed], geometry)
        expected = C0["sigma_gem_S_cm"][changed]*np.exp(-B*Hd[changed])
        homogeneity_error = float(np.max(abs(g/expected-1.), initial=0.))
        if homogeneity_error > CHECK_TOLERANCE: raise ValueError("Equal phase scaling lost GEM homogeneity")
        for k in result: result[k][changed] = corrected[k]
    return dict(version=VERSION, physical_parts=result, admitted=mask, changed=changed,
        effective_low_phase_S_cm=low, high_sigma_S_cm=high, rho_ohm_cm=rho,
        ordering_H=list(coordinates["ordering_H"]), damage_H=list(coordinates["damage_H"]),
        delta_vs_C0=result["prediction_log10_S_cm"]-C0["prediction_log10_S_cm"],
        delta_vs_original=result["prediction_log10_S_cm"]-old["prediction_log10_S_cm"],
        GEM_homogeneity_max_relative_error=homogeneity_error, scope=configuration()["scope"])


def empirical_response(coordinates, coefficients):
    """Sign-free SAME Ho/Hd response only; deliberately no physical certificate."""
    if type(coefficients) is not dict or set(coefficients) != set(CLOCKS): raise ValueError("Exactly ordering/damage coefficients required")
    co, cd = [_number(coefficients[k], k) for k in CLOCKS]
    n = len(coordinates["admitted"]); mask, Ho, Hd = _coordinates(coordinates, n)
    with np.errstate(over="raise", invalid="raise"):
        delta = co*Ho+cd*Hd
    if not np.isfinite(delta).all(): raise ValueError("Finite empirical response required")
    delta[~mask] = 0.
    return dict(correction_log10=delta, admitted=mask, scope="empirical SAME clocks, no GEM/phase/rho certificate")
