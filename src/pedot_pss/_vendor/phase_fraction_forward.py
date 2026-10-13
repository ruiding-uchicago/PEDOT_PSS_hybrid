"""Explicit row-f/fc effective state -> actual GEM -> frozen positive circuit.

No IO, fit/load, input admission, parameter guessing, ID routing or output patch.
Fractions are effective high-phase volume assumptions, NOT measured morphology.
Only TRAIN inverse/objective accept labels; caller enforces fold membership.
Do not reuse f-fixed inverse/offset/curvature with changed row fractions.
"""
import math
import numpy as np
from scipy.special import expit

try:
    from . import phase_connectivity_forward as connected
except ImportError:
    from . import phase_connectivity_forward as connected
network = connected.network
VERSION = "row-fraction-connectivity-effective-state-actual-forward-v1"
MIN_LOG_LOW_S_CM = connected.MIN_LOG_LOW_S_CM
CHECK_TOLERANCE = connected.CHECK_TOLERANCE
INVERSE_INTERVAL_TOLERANCE = connected.INVERSE_INTERVAL_TOLERANCE
LN10 = math.log(10.)


def phase_configuration(gem_model):
    """Frozen high conductivity/exponent/base geometry; no geometry is mutated."""
    return connected.phase_configuration(gem_model)


def _inputs(values, reference_parts, gem_model, fractions, thresholds, name):
    values, ref, config, fc = connected._inputs(values, reference_parts, gem_model, thresholds, name)
    f = connected._vector(fractions, len(values), "fractions")
    if ((f<0.) | (f>1.)).any(): raise ValueError("Require aligned0<=f<=1")
    return values, ref, config, f, fc


def _forward(state, ref, config, f, fc):
    lo, hi = config["min_log_low_S_cm"], config["max_log_low_S_cm"]
    if ((state<lo) | (state>hi)).any(): raise ValueError("Low-phase state outside bounds; no implicit clipping")
    low = 10.**state
    low[state==hi] = config["sigma_high_S_cm"]
    physics = network.transport._physics
    g = physics.gem_conductivity(low, config["sigma_high_S_cm"], f, fc, config["exponent"])
    out = network.network_with_rho(np.log10(g), ref["series_resistivity_ohm_cm"], ref["bypass_fraction"],
        anisotropy_factor=ref.get("anisotropy_factor"))
    actual = out["sigma_gem_S_cm"]
    residual = physics.gem_residual(actual, low, config["sigma_high_S_cm"], f, fc, config["exponent"])
    violation = ((actual<low*(1.-1e-11)) | (actual>config["sigma_high_S_cm"]*(1.+1e-11))).astype(float)
    if not np.isfinite(residual).all() or (abs(residual)>CHECK_TOLERANCE).any() or violation.any():
        raise ValueError("Actual GEM equation/phase bounds failed")
    out.update(gem_equation_residual=residual, gem_phase_bound_violation=violation,
        effective_low_phase_S_cm=low, phase_state_log_low_S_cm=state.copy(),
        phase_fraction_fraction=f.copy(), phase_fraction_threshold=fc.copy())
    return out


def _traces(out, ref):
    out.update({"phase_fraction_reference_"+key:value.copy() for key,value in ref.items()})
    for key,label in (("prediction_log10_S_cm","correction"),("gem_log10","gem_correction")):
        if key in ref: out["phase_fraction_"+label+"_from_reference_log10"] = out[key]-ref[key]
    return out


def solve_phase_state(log_low, reference_parts, gem_model, *, fractions, thresholds):
    """Query-only actual solve at explicit row f/fc; no target argument.

    f may cross fc and may equal0/1. Low state is log10(S/cm), bounded[-12,log10sh].
    Effective rho[ohm cm]/p/anisotropy are frozen; rho already includes scale and
    anisotropy and is NOT rescaled/clipped. Old traces live only under reference_.
    """
    state,ref,config,f,fc = _inputs(log_low,reference_parts,gem_model,fractions,thresholds,"low-phase state")
    return _traces(_forward(state,ref,config,f,fc),ref)


def forward_with_derivative(raw_scores, reference_parts, gem_model, *, fractions, thresholds, offset=0.):
    """Actual bounded state and dlog10(sigma)/draw using SAME explicit row f/fc.

    With A=(1-fc)/fc, l=(sl/G)^(1/t),h=(sh/G)^(1/t), U=(1-f)l/(l+A)^2,
    V=f*h/(h+A)^2, dlogG/dlogsl=U/(U+V). Multiply unchanged circuit and sigmoid
    slopes. Row f/fc are input-only frozen wrt raw; no df/draw term is implied.
    The scalar offset must be chosen on TRAIN only, then reused at query.
    """
    raw,ref,config,f,fc = _inputs(raw_scores,reference_parts,gem_model,fractions,thresholds,"raw scores")
    if not network._finite(offset): raise ValueError("One finite non-Boolean TRAIN offset required")
    with np.errstate(over="ignore"): shifted=raw+float(offset)
    if not np.isfinite(shifted).all(): raise ValueError("Finite raw+offset required")
    probability=expit(shifted);lo,hi=config["min_log_low_S_cm"],config["max_log_low_S_cm"]
    state=np.clip(lo+(hi-lo)*probability,lo,hi)
    out=_traces(_forward(state,ref,config,f,fc),ref)
    log_a=np.log((1.-fc)/fc);t=config["exponent"]
    log_l=(state-out["gem_log10"])*LN10/t;log_h=(hi-out["gem_log10"])*LN10/t
    with np.errstate(divide="ignore"):
        log_u=np.log1p(-f)+log_l-2.*np.logaddexp(log_l,log_a)
        log_v=np.log(f)+log_h-2.*np.logaddexp(log_h,log_a)
    gem_slope=np.exp(log_u-np.logaddexp(log_u,log_v))
    rho,p=out["series_resistivity_ohm_cm"],out["bypass_fraction"]
    with np.errstate(divide="ignore"):
        inverse_load=-np.logaddexp(0.,np.log(rho)+out["gem_log10"]*LN10)
        log_p,log_not_p=np.log(p),np.log1p(-p)
    circuit_slope=np.exp(np.logaddexp(log_p,log_not_p+2.*inverse_load)-np.logaddexp(log_p,log_not_p+inverse_load))
    state_slope=(hi-lo)*probability*expit(-shifted);derivative=gem_slope*circuit_slope*state_slope
    if (any(not np.isfinite(v).all() for v in (gem_slope,circuit_slope,state_slope,derivative)) or
            ((gem_slope<0.) | (gem_slope>1.+1e-12) | (circuit_slope<0.) | (circuit_slope>1.+1e-12) | (derivative<0.)).any()):
        raise ValueError("Finite nonnegative physical-chain derivatives required")
    out.update(phase_raw_score=raw.copy(),phase_raw_offset=np.full(len(raw),float(offset)),phase_sigmoid_probability=probability,
        dlog_gem_dlog_low=gem_slope,dlog_sigma_dlog_gem=circuit_slope,dlog_sigma_dlog_low=gem_slope*circuit_slope,
        dlog_low_draw=state_slope,dprediction_draw=derivative)
    return out


def train_target_inverse(y_train, reference_parts, gem_model, *, fractions, thresholds, iterations=60):
    """TRAIN-only monotone bisection, honest projection and original11 fields.

    Endpoints, all bisections and replay use SAME f/fc. f1 is forward-valid but
    output-flat wrt low state and explicitly rejected here, not fake-inverted.
    """
    y,ref,config,f,fc=_inputs(y_train,reference_parts,gem_model,fractions,thresholds,"training targets")
    if type(iterations) is not int or not 40<=iterations<=128: raise ValueError("40..128 integer bisection iterations required")
    lower=np.full(len(y),config["min_log_low_S_cm"]);upper=np.full(len(y),config["max_log_low_S_cm"])
    minimum,maximum=(_forward(v,ref,config,f,fc)["prediction_log10_S_cm"] for v in (lower,upper))
    if (maximum-minimum<=INVERSE_INTERVAL_TOLERANCE).any(): raise ValueError("Numerically distinguishable monotone interval required")
    below,above=y<minimum,y>maximum;projected=np.clip(y,minimum,maximum)
    lo,hi=lower.copy(),upper.copy()
    for _ in range(iterations):
        mid=lo+(hi-lo)/2.;left=_forward(mid,ref,config,f,fc)["prediction_log10_S_cm"]<projected
        lo[left]=mid[left];hi[~left]=mid[~left]
    state=lo+(hi-lo)/2.;state[projected==minimum]=lower[projected==minimum];state[projected==maximum]=upper[projected==maximum]
    replay=_forward(state,ref,config,f,fc)["prediction_log10_S_cm"]
    if (abs(replay-projected)>1e-9).any(): raise ValueError("Actual inverse replay exceeds1e-9")
    return dict(latent_log_low_S_cm=state,projected_target_log10_S_cm=projected,target_below_attainable_range=below,
        target_above_attainable_range=above,target_was_projected=below|above,target_within_attainable_range=~(below|above),
        projection_error_log10=projected-y,attainable_lower_log10_S_cm=minimum,attainable_upper_log10_S_cm=maximum,
        reconstructed_log10_S_cm=replay,inverse_replay_error_log10=replay-projected)


def training_squared_objective(raw_scores,y_train,reference_parts,gem_model,*,fractions,thresholds,offset=0.,hessian_floor=1e-6,sample_weight=None):
    """TRAIN squared loss: g=2w(P-y)P'; GN H=w*max(2P'^2,floor), NOT exact H.

    Half-square caller halves BOTH outputs; floor precedes weights, zero weights
    give exact0/0. Floor cannot restore a saturated zero gradient. No query labels.
    """
    out=forward_with_derivative(raw_scores,reference_parts,gem_model,fractions=fractions,thresholds=thresholds,offset=offset)
    n=len(out["prediction_log10_S_cm"]);y=connected._vector(y_train,n,"TRAIN labels")
    if not network._finite(hessian_floor) or hessian_floor<=0.: raise ValueError("Positive finite non-Boolean Hessian floor required")
    weights=np.ones(n) if sample_weight is None else connected._vector(sample_weight,n,"sample weights")
    if (weights<0.).any(): raise ValueError("Nonnegative sample weights required")
    derivative=out["dprediction_draw"]
    with np.errstate(over="ignore",invalid="ignore"):
        gradient=2.*weights*(out["prediction_log10_S_cm"]-y)*derivative
        hessian=weights*np.maximum(2.*derivative**2,float(hessian_floor))
    if not np.isfinite(gradient).all() or not np.isfinite(hessian).all(): raise ValueError("Finite loss gradient/GN Hessian required")
    return gradient,hessian
