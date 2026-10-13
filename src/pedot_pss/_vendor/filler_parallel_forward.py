"""Pure effective filler-path competition hypothesis; no IO, fit or routing.

sigma_new = sigma_host*exp(-b*x) + A*h(x;k),
h = (k*x/2)**2 * exp(2-k*x), k = exp(kappa0+kappa1*z).
x is a reported nominal MASS coordinate, never a converted volume fraction.
z is an explicitly admitted film-forming protocol class, not source identity.
A is an effective path amplitude (S/cm), not intrinsic filler conductivity;
b/k are not measured aggregation, energy-filtering barriers or morphology.
Caller owns input-only material/protocol admission and TRAIN-only calibration.

Conductivities are returned in log10(S/cm), so extreme finite host logs do not
overflow a linear conversion. Disabled filler branches are -inf (exact zero);
inactive branch/protocol/mass diagnostics are NaN (unknown/inapplicable).
This is not a unified GEM solve or a microscopic percolation law.
"""
from __future__ import annotations

import math
from numbers import Real
import numpy as np

VERSION = "effective-filler-parallel-forward-v1"
LN10 = math.log(10.)
PARAMETER_BOUNDS = {
    "b": (0., 12.), "A": (0., 1e6),
    "kappa0": (math.log(.5), math.log(200.)),
    "kappa1": (-math.log(100.), math.log(100.)),
}


def _parameter(value, name):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError("Explicit non-Boolean scalar parameter required: " + name)
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("Representable scalar parameter required: " + name) from exc
    lo, hi = PARAMETER_BOUNDS[name]
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError("Parameter outside predeclared bounds: " + name)
    return value


def _numeric(value, n, name, *, unknown=False):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (n,):
        raise ValueError("Aligned nonempty 1D numeric array required: " + name)
    for v in raw:
        if unknown and v is None:
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
            raise ValueError("Non-Boolean numeric entries required: " + name)
    try:
        out = np.asarray(value, dtype=float).copy()
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("Representable numeric entries required: " + name) from exc
    if np.isinf(out).any() or (not unknown and not np.isfinite(out).all()):
        raise ValueError("Finite entries required: " + name)
    return out


def _boolean(value, n, name):
    raw = np.asarray(value)
    if raw.shape != (n,) or raw.dtype.kind != "b":
        raise ValueError("Aligned strict Boolean array required: " + name)
    return raw.copy()


def _logadd10(a, b):
    with np.errstate(over="ignore", under="ignore"):
        return np.maximum(a, b) + np.log1p(np.exp(-np.abs(a-b)*LN10))/LN10


def forward(host_log10_sigma_S_cm, x, z, admission_mask, *, b, A, kappa0, kappa1):
    """Return positive branches/total as log10 conductivity and actual traces.

    All inputs are aligned nonempty 1D arrays; no scalar broadcasting. Host logs
    must be finite. Active x must be known in [0,1]; inactive x may be None/NaN,
    but supplied finite out-of-range values, infinities and booleans fail closed.
    z/mask are strict bool arrays. Inactive z=False is only a container value,
    NOT an assertion that an unknown protocol is spin coating: returned z is NaN.
    Parameters are validated even for a null or entirely inactive batch.
    Inactive, x=0 and A=b=0 predictions copy the host EXACTLY.
    """
    raw = np.asarray(host_log10_sigma_S_cm)
    if raw.ndim != 1 or not raw.size:
        raise ValueError("Nonempty 1D host conductivity required")
    n = raw.size
    host = _numeric(host_log10_sigma_S_cm, n, "host log10 conductivity")
    mass = _numeric(x, n, "nominal mass coordinate", unknown=True)
    protocol = _boolean(z, n, "protocol z")
    active = _boolean(admission_mask, n, "admission mask")
    if ((mass < 0.) | (mass > 1.)).any() or not np.isfinite(mass[active]).all():
        raise ValueError("Known active mass coordinate in [0,1] required")
    params = {name: _parameter(value, name) for name, value in
              (("b", b), ("A", A), ("kappa0", kappa0), ("kappa1", kappa1))}
    b, A, kappa0, kappa1 = (params[k] for k in ("b", "A", "kappa0", "kappa1"))
    prediction = host.copy()
    host_branch, filler_branch, shape, k, retention, mass_trace, z_trace = (
        np.full(n, np.nan) for _ in range(7))
    correction, host_correction, residual = (np.zeros(n) for _ in range(3))
    derivative = np.ones(n)
    mass_trace[active], z_trace[active] = mass[active], protocol[active].astype(float)
    log_k = kappa0 + kappa1*protocol[active].astype(float)
    k[active] = np.exp(log_k)
    host_correction[active] = -b*mass[active]/LN10
    host_branch[active] = host[active] + host_correction[active]
    retention[active] = np.exp(-b*mass[active])
    filler_branch[active], shape[active] = -np.inf, 0.
    positive = active & (mass > 0.)
    if positive.any():
        delta = kappa0+kappa1*protocol[positive].astype(float)+np.log(mass[positive])-math.log(2.)
        # log(h)=2*(delta-expm1(delta)); no k*x underflow or 0*inf tail.
        # min(0,...) only normalizes roundoff at the analytic maximum h=1.
        log_h = np.minimum(0., 2.*(delta-np.expm1(delta)))
        with np.errstate(under="ignore"):
            shape[positive] = np.exp(log_h)
        if A > 0.:
            filler_branch[positive] = (math.log(A)+log_h)/LN10
    changed = positive if A != 0. or b != 0. else np.zeros(n, dtype=bool)
    prediction[changed] = _logadd10(host_branch[changed], filler_branch[changed])
    correction[changed] = prediction[changed]-host[changed]
    with np.errstate(over="ignore", under="ignore"):
        host_share = np.exp((host_branch[active]-prediction[active])*LN10)
        filler_share = np.exp((filler_branch[active]-prediction[active])*LN10)
    derivative[active] = host_share
    residual[active] = host_share+filler_share-1.
    bounds = active & ((shape < 0.) | (shape > 1.) | (prediction < host_branch))
    if (not all(np.isfinite(v).all() for v in (prediction, correction, derivative, residual))
            or np.any((derivative < 0.) | (derivative > 1.)) or bounds.any()
            or np.max(np.abs(residual), initial=0.) > 1e-10):
        raise ValueError("Positive parallel equation or numeric bounds failed")
    return dict(prediction_log10_S_cm=prediction, host_log10_sigma_S_cm=host,
        host_branch_log10_S_cm=host_branch, filler_branch_log10_S_cm=filler_branch,
        total_correction_log10=correction, host_correction_log10=host_correction,
        host_retention_factor=retention, filler_shape=shape, effective_k=k,
        nominal_mass_fraction=mass_trace, protocol_z=z_trace, admission_mask=active,
        dlog_prediction_dlog_host=derivative, parallel_equation_residual=residual,
        positive_bound_violation=bounds, metadata=dict(version=VERSION, parameters=params,
        mass_role="reported_nominal_mass_coordinate_not_volume_fraction",
        state_role="effective_parallel_path_competition_not_identified_micromechanism",
        conductivity_units="log10(S/cm)", amplitude_units="S/cm"))
