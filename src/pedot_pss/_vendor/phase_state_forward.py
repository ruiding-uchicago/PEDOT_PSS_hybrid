"""Effective low-phase state -> actual GEM -> fixed positive reference circuit.

No files, fitting, model loading, ID routing or additive conductivity residual.
The learned state is NOT measured PSS conductivity or an identified microscopic
phase. Frozen rho already includes its clipping, anisotropy and interface scale.
Only train_target_inverse accepts labels, explicitly TRAINING labels; the caller
must enforce fold membership. Query prediction uses solve_phase_state only.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

try:
    from . import solvent_drying_boundary as network
except ImportError:
    from . import solvent_drying_boundary as network
VERSION = "effective-low-phase-physical-output-v1"
MIN_LOG_LOW_S_CM = -12.
CHECK_TOLERANCE = 1e-10
INVERSE_INTERVAL_TOLERANCE = 1e-12
NETWORK_KEYS = frozenset(("prediction_log10_S_cm", "gem_log10", "domain_correction_log10",
    "series_correction_log10", "sigma_domain_S_cm", "sigma_gem_S_cm", "sigma_series_S_cm",
    "series_resistivity_ohm_cm", "barrier_rho_ohm_cm", "anisotropy_factor", "bypass_fraction",
    "closure_residual", "domain_bound_violation", "gem_equation_residual", "gem_phase_bound_violation"))
ACID_TRACE_KEYS = frozenset("acid_process_"+key for key in (
    "protocol_admitted", "has_supported_concentration", "has_neat_proxy", "saturation", "hydration_factor", "gate",
    "is_admitted", "coordinate_changed", "old_coordinate", "new_coordinate", "base_prediction_log10_S_cm",
    "base_gem_log10", "correction_log10", "gem_correction_log10", "old_log_rho_linear", "new_log_rho_linear"))


def phase_configuration(gem_model):
    """Read actual frozen geometry only; neither sparse coefficients nor fit state."""
    if type(gem_model) is not network.transport._physics.GreyBoxTransport:
        raise TypeError("Actual GreyBoxTransport GEM geometry required")
    values={"sigma_high_S_cm":gem_model.sigma_high,"fraction":gem_model.fraction,
            "threshold":gem_model.threshold,"exponent":gem_model.exponent}
    if (any(not network._finite(v) for v in values.values()) or gem_model.sigma_high<=10.**MIN_LOG_LOW_S_CM or
            not 0.<=gem_model.fraction<gem_model.threshold<1. or gem_model.exponent<=0.):
        raise ValueError("Finite physical GEM phase geometry required")
    values={k:float(v) for k,v in values.items()}
    values.update(min_log_low_S_cm=MIN_LOG_LOW_S_CM,max_log_low_S_cm=math.log10(gem_model.sigma_high))
    return values


def _context(reference_parts,n):
    if not isinstance(reference_parts,Mapping) or not {"series_resistivity_ohm_cm","bypass_fraction"}.issubset(reference_parts):
        raise ValueError("Actual aligned reference rho and bypass required")
    if not set(reference_parts).issubset(NETWORK_KEYS|ACID_TRACE_KEYS):
        raise ValueError("Only actual physical/reference trace fields accepted; no targets/IDs")
    ref={key:network._array(value,n,key) for key,value in reference_parts.items()}
    rho,p=ref["series_resistivity_ohm_cm"],ref["bypass_fraction"]
    if (rho<0.).any() or ((p<0.)|(p>1.)).any() or ("anisotropy_factor" in ref and (ref["anisotropy_factor"]<=0.).any()):
        raise ValueError("Nonnegative reference rho, positive anisotropy and [0,1] bypass required")
    if "barrier_rho_ohm_cm" in ref and not np.array_equal(ref["barrier_rho_ohm_cm"],rho):
        raise ValueError("Reference effective rho aliases differ")
    for key in ("gem_equation_residual","closure_residual"):
        if key in ref and (abs(ref[key])>CHECK_TOLERANCE).any():raise ValueError("Reference physical equation failed")
    for key in ("gem_phase_bound_violation","domain_bound_violation"):
        if key in ref and ref[key].any():raise ValueError("Reference physical bounds failed")
    return ref


def _forward(log_low,ref,phase):
    if ((log_low<phase["min_log_low_S_cm"])|(log_low>phase["max_log_low_S_cm"])).any():
        raise ValueError("Low-phase log state outside declared bounds; no silent clipping")
    low=10.**log_low
    low[log_low==phase["max_log_low_S_cm"]]=phase["sigma_high_S_cm"]
    g=network.transport._physics.gem_conductivity(low,phase["sigma_high_S_cm"],phase["fraction"],phase["threshold"],phase["exponent"])
    result=network.network_with_rho(np.log10(g),ref["series_resistivity_ohm_cm"],ref["bypass_fraction"],
        anisotropy_factor=ref.get("anisotropy_factor"))
    actual_g=result["sigma_gem_S_cm"]
    residual=network.transport._physics.gem_residual(actual_g,low,phase["sigma_high_S_cm"],phase["fraction"],phase["threshold"],phase["exponent"])
    violation=((actual_g<low*(1.-1e-11))|(actual_g>phase["sigma_high_S_cm"]*(1.+1e-11))).astype(float)
    if not np.isfinite(residual).all() or (abs(residual)>CHECK_TOLERANCE).any() or violation.any():
        raise ValueError("Actual GEM equation/phase bounds failed")
    result.update(gem_equation_residual=residual,gem_phase_bound_violation=violation,
        effective_low_phase_S_cm=low,phase_state_log_low_S_cm=log_low.copy())
    return result


def solve_phase_state(log_low,reference_parts,gem_model):
    """Physical-output-only prediction from input-predicted effective log state.

    log_low is a finite aligned vector in [-12, log10(sh)] [log10(S/cm)].
    rho [ohm cm] and bypass are exactly frozen from the supplied actual P4
    forward; no new clipping, anisotropy multiplication or interface scaling.
    Old reference/gate traces are copied under phase_state_reference_* ONLY:
    they describe the earlier P4 state, not recomputed gate effects at this new
    state. New final/GEM/domain quantities and equation checks are actual solves.
    No coefficient from gem_model is changed or used to predict this new state.
    """
    raw=np.asarray(log_low)
    if raw.ndim!=1:raise ValueError("Aligned one-dimensional low-phase vector required")
    state=network._array(log_low,len(raw),"low-phase state")
    phase=phase_configuration(gem_model);ref=_context(reference_parts,len(state))
    result=_forward(state,ref,phase)
    result.update({"phase_state_reference_"+key:value.copy() for key,value in ref.items()})
    if "prediction_log10_S_cm" in ref:
        result["phase_state_correction_from_reference_log10"]=result["prediction_log10_S_cm"]-ref["prediction_log10_S_cm"]
    if "gem_log10" in ref:
        result["phase_state_gem_correction_from_reference_log10"]=result["gem_log10"]-ref["gem_log10"]
    return result


def train_target_inverse(y_train,reference_parts,gem_model,*,iterations=60):
    """TRAIN-only generated latent supervision, NOT measured/identified state.

    Bisection uses the actual monotone forward with fixed rho/p and GEM geometry.
    Out-of-range training targets are explicitly projected to attainable output
    endpoints; flags and signed projection error (projected - original) remain.
    Endpoints are allowed limiting effective states, not proof of equal material
    phases. The caller alone enforces training membership: this pure function
    cannot certify where its y_train came from. NEVER call on held-out targets.
    """
    raw=np.asarray(y_train)
    if raw.ndim!=1:raise ValueError("Aligned one-dimensional training targets required")
    y=network._array(y_train,len(raw),"training targets")
    if type(iterations) is not int or not 40<=iterations<=128:raise ValueError("40..128 integer bisection iterations required")
    phase=phase_configuration(gem_model);ref=_context(reference_parts,len(y))
    lower=np.full(len(y),phase["min_log_low_S_cm"]);upper=np.full(len(y),phase["max_log_low_S_cm"])
    minimum=_forward(lower,ref,phase)["prediction_log10_S_cm"]
    maximum=_forward(upper,ref,phase)["prediction_log10_S_cm"]
    if (maximum-minimum<=INVERSE_INTERVAL_TOLERANCE).any():raise ValueError("Numerically distinguishable monotone output interval required")
    below,above=y<minimum,y>maximum
    projected=np.clip(y,minimum,maximum)
    lo,hi=lower.copy(),upper.copy()
    for _ in range(iterations):
        mid=lo+(hi-lo)/2.
        predicted=_forward(mid,ref,phase)["prediction_log10_S_cm"]
        left=predicted<projected
        lo[left]=mid[left];hi[~left]=mid[~left]
    state=lo+(hi-lo)/2.
    state[projected==minimum]=lower[projected==minimum]
    state[projected==maximum]=upper[projected==maximum]
    replay=_forward(state,ref,phase)["prediction_log10_S_cm"]
    return dict(latent_log_low_S_cm=state,projected_target_log10_S_cm=projected,
        target_below_attainable_range=below,target_above_attainable_range=above,
        target_was_projected=below|above,target_within_attainable_range=~(below|above),
        projection_error_log10=projected-y,attainable_lower_log10_S_cm=minimum,
        attainable_upper_log10_S_cm=maximum,reconstructed_log10_S_cm=replay,
        inverse_replay_error_log10=replay-projected)
