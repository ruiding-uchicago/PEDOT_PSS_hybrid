"""Inference-only m-stack surrogate (1013): joint late-head combiner + prediction-time physics guards
around a fitted original M/P predictor (original_clean_surrogate.OriginalCleanSurrogate).

Self-contained for packaging: imports only numpy and the original model modules; no research helpers,
no sys.path changes, no file reads at import or predict time (the parent's frozen MatBERT encoder is
only used when field vectors are not supplied, exactly as in the parent's own predict_values).
Vendoring needs only the module-level imports below turned into relative imports.

Forward per row (no labels accepted):
  parent base -> parent special-physics chain (acid/salt/PEO, fitted) -> guard rules (supports stored as
  data, computed from the deployment TRAIN rows when the object was built) -> parent filler/DAE ->
  the parent's 7 late stages evaluated at the parent's own sequential state (identical to the parent's
  forward) -> M = state_before_late + b + sum_s w_s * delta_s on columns 0 and 3 (frozen joint weights)
  -> P = original physics adapter with the stored (arm-refitted) P state. Platform (Polybot) rows have
  no late block and no admitted branch, so they are returned exactly as the parent returns them.
"""
from __future__ import annotations

import numpy as np

from . import original_clean_surrogate as ocs
from . import original_special_physics as osp
from . import original_filler_dae as ofd
from . import acid_interface_closure as acid
from .clean_inputs import canonicalize

VERSION = "m-stack-joint-late-combiner-infer-v1-1013"
STAGE_KINDS = ("global_kernel",) + tuple(ocs.LATE_HEADS)


# ----------------------------------------------------------------------------- data objects
class FrozenLinearCombiner:
    """delta = b + D @ w with fixed weights (D: n x 7 stage deltas)."""

    def __init__(self, weights, intercept=0.0, description=None):
        self.weights = np.asarray(weights, dtype=float).copy()
        self.intercept = float(intercept)
        self.description = dict(description or {})
        if self.weights.shape != (len(STAGE_KINDS),) or not np.isfinite(self.weights).all():
            raise ValueError("seven finite late-stage weights required")

    def predict_delta(self, D, S=None):
        return self.intercept + np.asarray(D, dtype=float) @ self.weights

    def main_weights(self):
        return self.weights.copy()


class GuardSpec:
    """Prediction-time physics-branch guard rules (thresholds/supports are data; no labels).

    peo: "saved" (the fitted guarded PEO expert as is) | "host1" (phi == 1.0, the pure host, is inside the
         fitted law's phi support) | "lower" (only phi below the support minimum is outside) | "off".
    peo_outside: "total" (outside rows get [parent total]*3 + [0, 0], the saved behaviour) | "state" (outside
         rows keep the complete parent state).
    acid / salt / filler: "keep" | "support" (outside the stored TRAIN coordinate range -> parent state) |
         "support_id" (acid: identity-specific range) | "off".
    cap: largest |change of the total| an acid/salt/filler increment may apply (None = no cap).
    """

    def __init__(self, peo="saved", peo_outside="total", acid="keep", salt="keep", filler="keep", cap=None,
                 name=None):
        self.peo, self.peo_outside, self.acid, self.salt, self.filler = peo, peo_outside, acid, salt, filler
        self.cap = None if cap is None else float(cap)
        self.name = name

    def describe(self):
        return dict(name=self.name, peo=self.peo, peo_outside=self.peo_outside, acid=self.acid, salt=self.salt,
                    filler=self.filler, cap=self.cap)

    def is_identity(self):
        return (self.peo == "saved" and self.peo_outside == "total" and self.acid == "keep" and self.salt == "keep"
                and self.filler == "keep" and self.cap is None)


# ----------------------------------------------------------------------------- branch admission (input only)
def branch_inputs(raw_rows):
    """Admission masks/coordinates of the original physics branches for raw (original_view) rows."""
    chain = [canonicalize(r) for r in raw_rows]
    ai = osp.acid_inputs(chain)
    si = osp.salt_inputs(chain)
    pi = osp.peo_inputs(chain)
    fi = ofd.input_admission(raw_rows)
    return {
        "acid_adm": np.asarray(ai["protocol_admitted"], bool),
        "acid_conc": np.array([r.get(acid.CONCENTRATION_KEY, np.nan) if r else np.nan for r in ai["acid_rows"]], float),
        "acid_id": np.array([str(r.get(acid.IDENTITY_KEY, "")) if r else "" for r in ai["acid_rows"]]),
        "salt_adm": np.asarray(si["coordinates"]["active"], bool),
        "salt_coord": np.column_stack([si["coordinates"][k] for k in ("concentration_mol_L", "cation_softness",
                                                                      "contact_temperature_K", "contact_seconds")]),
        "peo_adm": np.asarray(pi["active"], bool), "peo_phi": np.asarray(pi["phi"], float),
        "peo_family": np.asarray(pi["family"], int),
        "filler_adm": np.asarray(fi["admission_mask"], bool), "filler_x": np.asarray(fi["x"], float),
        "filler_z": np.asarray(fi["z"], bool), "dae_adm": np.asarray(fi["dae_mask"], bool),
        "chain_rows": chain,
    }


def _unguarded_peo_predict(expert):
    """The fitted PEO law without any support guard: the original expert's predict (found in the class
    hierarchy, so a monkeypatched module-level name cannot change it)."""
    for cls in type(expert).__mro__:
        if cls.__name__ == "OriginalPEOExpert" and cls.__module__.split(".")[-1] == "original_special_physics":
            return lambda rows, parent: cls.predict(expert, rows, parent)
    raise TypeError("PEO expert does not derive from original_special_physics.OriginalPEOExpert")


def _peo_law_rows(expert, adm):
    if getattr(expert, "cross_fit_", None) is not None:
        return adm["peo_adm"].copy()
    if getattr(expert, "old_fit_", None) is not None:
        return adm["peo_adm"] & (adm["peo_family"] == 0)
    return np.zeros(len(adm["peo_adm"]), bool)


def apply_special_guard(spec, model, adm, base_state, special_state, supports):
    """Guarded after-special state (n x 5) built from the parent chain's output."""
    out = special_state.copy()
    n = len(out)
    expert = model.special_.peo_
    law_rows = _peo_law_rows(expert, adm)
    moved = np.any(np.abs(special_state - base_state) > 0, axis=1)
    acid_rows = adm["acid_adm"] & moved & ~law_rows
    salt_rows = adm["salt_adm"] & moved & ~law_rows
    if np.any(acid_rows & salt_rows):
        raise ValueError("acid and salt admitted on the same row; per-branch guard undefined")
    if law_rows.any() and not (spec.peo == "saved" and spec.peo_outside == "total"):
        phi = adm["peo_phi"]
        sup = getattr(expert, "phi_support_", None)
        q_law = _unguarded_peo_predict(expert)(adm["chain_rows"], base_state[:, 0])
        if spec.peo == "off":
            outside = law_rows.copy()
        elif sup is None:
            outside = np.zeros(n, bool)
        else:
            with np.errstate(invalid="ignore"):
                below, above = phi < sup[0], phi > sup[1]
            if spec.peo == "saved":
                outside = law_rows & (below | above)
            elif spec.peo == "host1":
                outside = law_rows & (below | (above & ~(phi == 1.0)))
            elif spec.peo == "lower":
                outside = law_rows & below
            else:
                raise ValueError(spec.peo)
        inside = law_rows & ~outside
        out[inside, :3] = q_law[inside, None]
        out[inside, 3:] = 0.0
        if spec.peo_outside == "total" and spec.peo != "off":
            out[outside, :3] = base_state[outside, 0][:, None]
            out[outside, 3:] = 0.0
        else:
            out[outside] = base_state[outside]
    if spec.acid != "keep" and acid_rows.any():
        if spec.acid == "off":
            g = acid_rows
        elif spec.acid == "support":
            lo, hi = supports["acid_conc"] if supports["acid_conc"] else (np.inf, -np.inf)
            c = adm["acid_conc"]
            g = acid_rows & ((c < lo) | (c > hi))
        elif spec.acid == "support_id":
            g = np.zeros(n, bool)
            for i in np.flatnonzero(acid_rows):
                r = supports["acid_by_id"].get(adm["acid_id"][i])
                g[i] = r is None or not (r[0] <= adm["acid_conc"][i] <= r[1])
        else:
            raise ValueError(spec.acid)
        out[g] = base_state[g]
    if spec.salt != "keep" and salt_rows.any():
        if spec.salt == "off":
            g = salt_rows
        elif spec.salt == "support":
            box = supports["salt_box"]
            if box is None:
                g = salt_rows
            else:
                lo, hi = np.asarray(box[0]), np.asarray(box[1])
                c = adm["salt_coord"]
                g = salt_rows & np.any((c < lo) | (c > hi), axis=1)
        else:
            raise ValueError(spec.salt)
        out[g] = base_state[g]
    if spec.cap is not None:
        g = (acid_rows | salt_rows) & (np.abs(out[:, 0] - base_state[:, 0]) > spec.cap)
        out[g] = base_state[g]
    return out


def apply_filler_guard(spec, adm, special_state, filler_state, supports):
    out = filler_state.copy()
    rows = adm["filler_adm"] & np.any(np.abs(filler_state - special_state) > 0, axis=1)
    if spec.filler != "keep" and rows.any():
        if spec.filler == "off":
            g = rows
        elif spec.filler == "support":
            g = np.zeros(len(out), bool)
            for i in np.flatnonzero(rows):
                r = supports["filler"].get(int(adm["filler_z"][i]))
                g[i] = r is None or not (r[0] <= adm["filler_x"][i] <= r[1])
        else:
            raise ValueError(spec.filler)
        out[g] = special_state[g]
    if spec.cap is not None:
        g = rows & (np.abs(out[:, 0] - special_state[:, 0]) > spec.cap)
        out[g] = special_state[g]
    return out


# ----------------------------------------------------------------------------- stage forward (parent's arithmetic)
def upstream(model, clean, raw):
    base = model.base_.predict_values(clean)
    special = ocs._values(model.special_.predict_values(raw, base.copy())) if model.special_ is not None else base.copy()
    filler = ocs._values(model.filler_.predict_values(raw, special.copy())) if model.filler_ is not None else special.copy()
    return base, special, filler


def stages_needed(combiner, stages):
    """Number of leading late stages that can change the output.

    A late stage only reaches the prediction through its combiner weight and through the sequential parent
    state read by the stages after it. Stages after the last one with a non-zero combiner weight therefore
    cannot change the output; their deltas are not computed (0 is used, so weight x delta is exactly 0).
    """
    w = np.asarray(combiner.main_weights(), float)
    if len(w) != len(stages):
        return len(stages)
    nonzero = np.flatnonzero(w != 0.0)
    return int(nonzero.max()) + 1 if nonzero.size else 0


def late_deltas(model, raw, science, field_vectors, state_before_late, active, n_needed=None):
    """The 7 late-stage deltas at the parent's sequential state (n_stage x n, NaN on platform rows).

    Stages with index >= n_needed are not evaluated (delta 0); see stages_needed.
    """
    values = np.asarray(state_before_late, float).copy()
    stages = model.stages_
    deltas = np.full((len(stages), len(values)), np.nan)
    selected = np.flatnonzero(active)
    if len(selected):
        sci = [science[i] for i in selected]
        rw = [raw[i] for i in selected]
        for s, st in enumerate(stages):
            if n_needed is not None and s >= n_needed:
                deltas[s, selected] = 0.0
                continue
            if st["kind"] == "global_kernel":
                d = st["state"].predict_delta(sci, values[selected][:, ocs.PARENT4])
            else:
                d = ocs.predict_delta(st["state"], rw, sci, values[selected][:, ocs.PARENT4],
                                      field_vectors=np.asarray(field_vectors)[selected] if st["kind"] == "matbert" else None)
            deltas[s, selected] = d
            if not st["weight"]:
                continue
            values[selected, 0] += st["weight"] * d
            values[selected, 3] += st["weight"] * d
    return deltas


def combine(state_before_late, deltas, active, combiner):
    M = np.asarray(state_before_late, float).copy()
    sel = np.flatnonzero(active)
    if len(sel):
        d = combiner.predict_delta(deltas[:, sel].T, M[sel])
        M[sel, 0] += d
        M[sel, 3] += d
    ocs._closure(M)
    return M


def physics_P(physics_state, raw, M_state, active):
    P = np.asarray(M_state, float).copy()
    selected = np.flatnonzero(active)
    if len(selected):
        P[selected], _ = ocs.predict_original_physics(physics_state, [raw[i] for i in selected], P[selected])
    ocs._closure(P)
    return P


# ----------------------------------------------------------------------------- deployable predictor
class StackedSurrogate:
    """Fitted original M/P predictor + frozen joint late combiner + guard rules + (refitted) P state."""

    version = VERSION

    def __init__(self, parent, combiner, guard, supports, p_state=None, prefer_physics=None, info=None):
        if not isinstance(parent, ocs.OriginalCleanSurrogate):
            raise TypeError("parent must be a fitted original_clean_surrogate.OriginalCleanSurrogate")
        self.parent = parent
        self.combiner = combiner
        self.guard = guard
        self.supports = supports
        self.p_state = p_state if p_state is not None else parent.physics_
        self.prefer_physics = parent.prefer_physics if prefer_physics is None else bool(prefer_physics)
        self.info = dict(info or {})

    def forward(self, X, *, field_vectors=None, scientific=None):
        """dict with M and P (n x 5), the guarded state before the late block and the 7 stage deltas."""
        model = self.parent
        clean, raw, science, active = model._project(list(X), scientific)
        if not raw:
            e = np.empty((0, 5))
            return {"M": e, "P": e.copy(), "state_before_late": e.copy(),
                    "deltas": np.empty((len(STAGE_KINDS), 0)), "active": active}
        base, special, filler = upstream(model, clean, raw)
        if self.guard is not None and not self.guard.is_identity():
            adm = branch_inputs(raw)
            special_g = apply_special_guard(self.guard, model, adm, base, special, self.supports)
            if np.array_equal(special_g, special):
                filler_g = filler
            else:
                filler_g = ocs._values(model.filler_.predict_values(raw, special_g.copy()))
            filler_g = apply_filler_guard(self.guard, adm, special_g, filler_g, self.supports)
        else:
            filler_g = filler
        n_needed = stages_needed(self.combiner, model.stages_)
        needs_text_encoder = any(st["kind"] == "matbert" for st in model.stages_[:n_needed])
        if field_vectors is None and needs_text_encoder and np.any(active):
            field_vectors = model.encoder_.encode(raw)
        deltas = late_deltas(model, raw, science, field_vectors, filler_g, active, n_needed=n_needed)
        M = combine(filler_g, deltas, active, self.combiner)
        P = physics_P(self.p_state, raw, M, active)
        return {"M": M, "P": P, "state_before_late": filler_g, "deltas": deltas, "active": active}

    def predict_values(self, X, *, field_vectors=None, scientific=None, prefer_physics=None):
        out = self.forward(X, field_vectors=field_vectors, scientific=scientific)
        choose_P = self.prefer_physics if prefer_physics is None else prefer_physics
        return out["P"] if choose_P else out["M"]

    def predict(self, X, *, field_vectors=None, scientific=None):
        return self.predict_values(X, field_vectors=field_vectors, scientific=scientific)[:, 0]

    def predict_conductivity(self, X, **kw):
        return np.power(10.0, self.predict(X, **kw))

    def summary(self):
        return {"version": VERSION, "late_weights": dict(zip(STAGE_KINDS, self.combiner.main_weights().tolist())),
                "late_intercept": float(getattr(self.combiner, "intercept", 0.0)),
                "combiner": self.combiner.description,
                "guard": self.guard.describe() if self.guard is not None else None,
                "supports": self.supports, "prefer_physics_default": self.prefer_physics, **self.info}
