"""TRAIN-profiled alpha1 thermal experts; pure RAM, no data/IO or ancestor fit.

P and E have two extra parameters, not equal effective functional capacity.
The inherited M residual is a retrospective nuisance, not measured bulk sigma.
P's bounded multistart search is not a global-optimality proof. Only its actual
GEM/circuit parts have physical certificates; E's correction is residual-only.
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares

from . import thermal_competition_forward as forward
VERSION = "profiled-alpha1-two-clock-experts-v1"
ARMS = ("C0", "P", "E")
N_COMMON, MIN_TRAIN, ALPHA = 14, 16, 10.
STARTS = ((0., 0.), (2., 2.), (4., 0.), (0., 4.))
CORNERS = ((0., 0.), (4., 0.), (0., 4.), (4., 4.))
OPTIMIZER = dict(max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
FIT_KEYS = {"version", "arm", "beta", "intercept", "theta", "diagnostics"}
DIAGNOSTIC_KEYS = {"n_train", "n_common", "TRAIN_SSE", "penalized_objective",
    "residual_mean", "nuisance_gradient_max_abs", "intercept_gradient_abs", "nuisance_gap_bound",
    "clock_singular_values", "clock_rank", "profile_linear_solves", "quadratic_linear_solves", "physical_forward_evaluations",
    "optimizer_attempts", "optimizer_nfev", "selected_candidate", "candidates", "E_quadratic",
    "parameter_gradient", "projected_gradient", "bound_flags"}


def configuration():
    return dict(version=VERSION, arms=list(ARMS), n_common=N_COMMON, minimum_TRAIN=MIN_TRAIN,
        ridge_alpha_sum_loss=ALPHA, intercept_penalized=False, P_starts=[list(x) for x in STARTS],
        P_corners=[list(x) for x in CORNERS], P_optimizer=dict(OPTIMIZER),
        parameter_bounds=dict(C0=[0., 0.], P=[0., 4.], E=[-4., 4.]),
        E_solver="exact profiled box quadratic: free solution, edges, corners, exact null",
        forward=forward.configuration(), scope=__doc__.strip())


def _array(value, shape, name):
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "iuf" or
            any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat)):
        raise ValueError("Aligned nonboolean numeric array required: " + name)
    out = raw.astype(np.float64, copy=True)
    if not np.isfinite(out).all(): raise ValueError("Finite array required: " + name)
    return out


def _finite(value):
    if type(value) is dict: return all(type(k) is str and _finite(v) for k, v in value.items())
    if type(value) is list: return all(_finite(v) for v in value)
    if value is None or type(value) in (str, bool, int): return True
    return type(value) is float and math.isfinite(value)


def _inputs(bundle, coordinates, common, residual):
    n, _, _ = forward._bundle(bundle)
    admitted, Ho, Hd = forward._coordinates(coordinates, n)
    if not admitted.all(): raise ValueError("All expert input rows must be admitted")
    return _array(common, (n, N_COMMON), "common"), _array(residual, (n,), "rM"), np.column_stack((Ho, Hd))


def _theta(value, arm):
    if type(arm) is not str or arm not in ARMS: raise ValueError("Declared expert arm required")
    theta = _array(value, (2,), "theta"); lo, hi = (-4., 4.) if arm == "E" else (0., 4.)
    if np.any((theta < lo) | (theta > hi)) or (arm == "C0" and theta.any()):
        raise ValueError("Expert parameter bounds/null differ")
    return theta


def _parts(bundle, coordinates, theta, arm):
    A, B = theta if arm == "P" else (0., 0.)
    return forward.apply_competition(bundle, coordinates, A=float(A), B=float(B))["physical_parts"]


def _profiler(X, y_minus_rM):
    mean = X.mean(0); centered = X-mean
    with np.errstate(over="raise", invalid="raise"):
        gram = centered.T@centered+ALPHA*np.eye(N_COMMON)
    if not np.isfinite(gram).all(): raise ValueError("Unrepresentable nuisance Gram")
    factor = cho_factor(gram, lower=True, check_finite=True); count = 0

    def profile(physical):
        nonlocal count
        physical = _array(physical, y_minus_rM.shape, "profile physical")
        with np.errstate(over="raise", invalid="raise"):
            target = y_minus_rM-physical; center = float(target.mean())
            beta = cho_solve(factor, centered.T@(target-center), check_finite=True)
            intercept = float(center-mean@beta); prediction = X@beta+intercept
            error = target-prediction; gradient = X.T@error-ALPHA*beta
            sse = float(error@error); objective = float(sse+ALPHA*(beta@beta))
            gap = float(gradient@gradient/ALPHA)
        count += 1
        numbers = [*beta, intercept, sse, objective, gap, float(error.sum())]
        if (not np.isfinite(numbers).all() or gap > 1e-8*(1.+objective) or
                abs(error.sum()) > 1e-7*(1.+len(error))):
            raise ValueError("Finite converged nuisance normal equations required")
        return dict(beta=beta, intercept=intercept, error=error, objective=objective, sse=sse,
            residual_mean=float(error.mean()), nuisance_gradient_max_abs=float(np.max(abs(gradient))),
            intercept_gradient_abs=float(abs(error.sum())), nuisance_gap_bound=gap)

    return profile, centered, factor, lambda: count


def _box_candidates(Q, g):
    """PSD two-dimensional quadratic; enumerate every face, including rank loss."""
    Q = _array(Q, (2, 2), "quadratic Q"); g = _array(g, (2,), "quadratic g")
    if not np.allclose(Q, Q.T, atol=1e-10, rtol=1e-12): raise ValueError("Symmetric quadratic required")
    Q = (Q+Q.T)/2.; scale = max(1., float(np.max(abs(Q))))
    eigen = np.linalg.eigvalsh(Q)
    if eigen.min() < -1e-10*scale: raise ValueError("Positive-semidefinite profile required")
    candidates = [np.zeros(2)]
    free = np.linalg.lstsq(Q, g, rcond=None)[0]
    if (np.max(abs(Q@free-g), initial=0.) <= 1e-9*(1.+np.max(abs(g), initial=0.)) and
            np.all((free >= -4.) & (free <= 4.))): candidates.append(free)
    for fixed in range(2):
        other = 1-fixed
        for edge in (-4., 4.):
            z = np.zeros(2); z[fixed] = edge
            slope = g[other]-Q[other, fixed]*edge
            z[other] = np.clip(slope/Q[other, other], -4., 4.) if Q[other, other] > 0. else (4. if slope > 0. else -4. if slope < 0. else 0.)
            candidates.append(z)
    candidates.extend(np.array([a, b]) for a in (-4., 4.) for b in (-4., 4.))
    return candidates


def fit_arm(bundle, coordinates, common_train, rM_train, y_train, arm):
    """One supervised fitting problem; only TRAIN arrays. No files/ancestor fit."""
    theta0 = _theta([0., 0.], arm); X, rM, H = _inputs(bundle, coordinates, common_train, rM_train)
    y = _array(y_train, (len(X),), "TRAIN y")
    if len(y) < MIN_TRAIN: raise ValueError("At least16 admitted TRAIN rows required")
    singular = np.linalg.svd(H-H.mean(0), compute_uv=False)
    rank = int(np.sum(singular > 1e-10*max(1., float(singular[0]))))
    if rank != 2: raise ValueError("Two nonsaturated independent TRAIN clock directions required")
    profile, centered, factor, solve_count = _profiler(X, y-rM)
    forward_count = 0; candidates = []; states = []

    def physical(theta):
        nonlocal forward_count
        parts = _parts(bundle, coordinates, theta, arm); forward_count += 1
        value = parts["prediction_log10_S_cm"]
        if arm == "E": value = value+H@theta
        return value

    def consider(theta, kind, status=0, success=True, nfev=0, message="exact candidate"):
        theta = _theta(theta, arm); state = profile(physical(theta)); states.append((theta.copy(), state))
        candidates.append(dict(kind=kind, theta=theta.tolist(), objective=state["objective"],
            TRAIN_SSE=state["sse"], status=int(status), success=bool(success), nfev=int(nfev), message=str(message)))

    E_quadratic = None; parameter_gradient = None; projected_gradient = None
    if arm == "P":
        for corner in CORNERS: consider(corner, "exact_corner")
        for start in STARTS:
            def residual(theta):
                state = profile(physical(theta))
                return np.r_[state["error"], math.sqrt(ALPHA)*state["beta"]]
            result = least_squares(residual, start, bounds=(0., 4.), **OPTIMIZER)
            consider(result.x, "bounded_least_squares", result.status, result.success, result.nfev, result.message)
    elif arm == "E":
        p0 = _parts(bundle, coordinates, theta0, arm)["prediction_log10_S_cm"]; forward_count += 1
        Z = H-H.mean(0); target = y-rM-p0; target -= target.mean(); cross = centered.T@Z
        Q = Z.T@Z-cross.T@cho_solve(factor, cross)
        g = Z.T@target-cross.T@cho_solve(factor, centered.T@target)
        E_quadratic = dict(Q=Q.tolist(), g=g.tolist())
        for theta in _box_candidates(Q, g): consider(theta, "exact_box_face")
    else: consider(theta0, "exact_null")
    best = min(range(len(states)), key=lambda i: states[i][1]["objective"])
    theta, selected = states[best]
    if arm == "E":
        parameter_gradient = 2.*(Q@theta-g)
        projected_gradient = theta-np.clip(theta-parameter_gradient, -4., 4.)
        if np.max(abs(projected_gradient)) > 1e-7*(1.+np.max(abs(g))): raise ValueError("Exact box projected normal equations failed")
        parameter_gradient, projected_gradient = parameter_gradient.tolist(), projected_gradient.tolist()
    diag = dict(n_train=len(y), n_common=N_COMMON, TRAIN_SSE=selected["sse"], penalized_objective=selected["objective"],
        **{k:selected[k] for k in ("residual_mean", "nuisance_gradient_max_abs", "intercept_gradient_abs", "nuisance_gap_bound")},
        clock_singular_values=singular.tolist(), clock_rank=rank, profile_linear_solves=solve_count(), quadratic_linear_solves=2 if arm == "E" else 0,
        physical_forward_evaluations=forward_count, optimizer_attempts=4 if arm == "P" else 0,
        optimizer_nfev=sum(c["nfev"] for c in candidates), selected_candidate=best, candidates=candidates,
        E_quadratic=E_quadratic, parameter_gradient=parameter_gradient, projected_gradient=projected_gradient,
        bound_flags=[bool(v in ((-4., 4.) if arm == "E" else (0., 4.))) for v in theta])
    fit = dict(version=VERSION, arm=arm, beta=selected["beta"].tolist(), intercept=selected["intercept"],
        theta=theta.tolist(), diagnostics=diag)
    if not _finite(fit): raise ValueError("Finite plain fit required")
    return fit


def _fit(fit):
    if type(fit) is not dict or set(fit) != FIT_KEYS or fit["version"] != VERSION or not _finite(fit):
        raise ValueError("Closed finite selected fit required")
    theta = _theta(fit["theta"], fit["arm"]); beta = _array(fit["beta"], (N_COMMON,), "saved beta")
    intercept = forward._number(fit["intercept"], "saved intercept")
    diag = fit["diagnostics"]
    if type(diag) is not dict or set(diag) != DIAGNOSTIC_KEYS: raise ValueError("Closed TRAIN diagnostics required")
    for key in ("n_train", "n_common", "clock_rank", "profile_linear_solves", "quadratic_linear_solves", "physical_forward_evaluations",
                "optimizer_attempts", "optimizer_nfev", "selected_candidate"):
        if type(diag[key]) is not int or diag[key] < 0: raise ValueError("Strict diagnostic count required")
    if diag["n_train"] < MIN_TRAIN or diag["n_common"] != N_COMMON or diag["clock_rank"] != 2:
        raise ValueError("TRAIN diagnostic support differs")
    for key in ("TRAIN_SSE", "penalized_objective", "nuisance_gradient_max_abs", "intercept_gradient_abs", "nuisance_gap_bound"):
        forward._number(diag[key], key, lower=0.)
    forward._number(diag["residual_mean"], "residual mean")
    if np.any(_array(diag["clock_singular_values"], (2,), "clock singular values") <= 0.): raise ValueError("Positive rank2 singular values required")
    candidates = diag["candidates"]
    if type(candidates) is not list or not candidates or diag["selected_candidate"] >= len(candidates): raise ValueError("Selected finite candidate required")
    for candidate in candidates:
        if (type(candidate) is not dict or set(candidate) != {"kind", "theta", "objective", "TRAIN_SSE", "status", "success", "nfev", "message"} or
                type(candidate["success"]) is not bool or type(candidate["status"]) is not int or
                type(candidate["nfev"]) is not int or candidate["nfev"] < 0 or
                any(type(candidate[k]) is not str for k in ("kind", "message"))): raise ValueError("Closed typed candidate required")
        _theta(candidate["theta"], fit["arm"])
        for key in ("objective", "TRAIN_SSE"): forward._number(candidate[key], key, lower=0.)
    chosen = candidates[diag["selected_candidate"]]
    if (not np.array_equal(chosen["theta"], theta) or chosen["objective"] != diag["penalized_objective"] or
            chosen["TRAIN_SSE"] != diag["TRAIN_SSE"] or diag["optimizer_nfev"] != sum(c["nfev"] for c in candidates) or
            diag["optimizer_attempts"] != (4 if fit["arm"] == "P" else 0) or
            diag["quadratic_linear_solves"] != (2 if fit["arm"] == "E" else 0)):
        raise ValueError("Selected parameters/cost/accounting differ")
    if type(diag["bound_flags"]) is not list or len(diag["bound_flags"]) != 2 or any(type(v) is not bool for v in diag["bound_flags"]):
        raise ValueError("Boolean bound flags required")
    if fit["arm"] == "E":
        q = diag["E_quadratic"]
        if type(q) is not dict or set(q) != {"Q", "g"}: raise ValueError("Exact empirical profile quadratic required")
        _array(q["Q"], (2, 2), "saved Q"); _array(q["g"], (2,), "saved g")
        for key in ("parameter_gradient", "projected_gradient"): _array(diag[key], (2,), key)
    elif any(diag[k] is not None for k in ("E_quadratic", "parameter_gradient", "projected_gradient")):
        raise ValueError("No empirical certificate on physical/null arm")
    return theta, beta, intercept


def predict_arm(bundle, coordinates, common_query, rM_query, fit):
    """No target, fitting, optimizer or IO; QUERY nuisance is selected TRAIN-only."""
    theta, beta, intercept = _fit(fit)
    X, rM, H = _inputs(bundle, coordinates, common_query, rM_query)
    parts = _parts(bundle, coordinates, theta, fit["arm"])
    with np.errstate(over="raise", invalid="raise"):
        residual = rM+X@beta+intercept
        if fit["arm"] == "E": residual += H@theta
        physical = parts["prediction_log10_S_cm"]
        values = np.column_stack((physical+residual, physical, physical, residual, parts["domain_correction_log10"]))
    return _array(values, (len(X), 5), "expert values"), parts
