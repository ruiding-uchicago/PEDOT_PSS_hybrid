"""PEO percolation expert with a TRAIN phi-support guard (m-refit, 1013).

``GuardedPEOExpert(OriginalPEOExpert)``

* fit: the unchanged parent fit, plus a record of the nominal-phi support
  ``phi_support_ = (min, max)`` of the admitted TRAIN rows that the fitted law
  actually used: all admitted rows when the cross-host (SHARED_MORPH) law is
  fitted, admitted family-0 rows when only the old percolation law is fitted,
  ``None`` when no law was fitted (the parent then returns the parent anyway).
* predict: the parent prediction, except that active (admitted) rows whose phi
  lies outside ``phi_support_`` get the supplied parent prediction back
  unchanged, so the fitted law (e.g. a percolation threshold) is never
  extrapolated below / above the compositions it was fitted on.

Install by ``original_special_physics.OriginalPEOExpert = GuardedPEOExpert``;
``OriginalSpecialPhysicsChain.fit`` resolves that module-level name at call time.
Only TRAIN labels enter fit; predict accepts no labels.
"""
from __future__ import annotations

import numpy as np

from . import original_special_physics as osp

GUARD_VERSION = "peo-phi-support-guard-v1-1013"


class GuardedPEOExpert(osp.OriginalPEOExpert):
    """OriginalPEOExpert whose fitted law is applied only inside its TRAIN phi support."""

    guard_version = GUARD_VERSION

    def _law_rows(self, inputs):
        """Admitted rows the fitted law uses (same branch order as the parent predict)."""
        if getattr(self, "cross_fit_", None) is not None:
            return np.asarray(inputs["active"], bool).copy()
        if getattr(self, "old_fit_", None) is not None:
            return np.asarray(inputs["active"], bool) & (np.asarray(inputs["family"]) == 0)
        return np.zeros(len(inputs["active"]), bool)

    def fit(self, rows, y_train):
        rows = list(rows)
        super().fit(rows, y_train)
        inputs = osp.peo_inputs(rows)
        used = self._law_rows(inputs)
        phi = np.asarray(inputs["phi"], float)[used]
        if used.any():
            if not np.isfinite(phi).all():
                raise ValueError("Admitted PEO TRAIN rows must have finite nominal phi")
            self.phi_support_ = (float(np.min(phi)), float(np.max(phi)))
        else:
            self.phi_support_ = None
        self.n_law_train_ = int(used.sum())
        self.law_kind_ = ("cross_host_" + self.cross_arm if getattr(self, "cross_fit_", None) is not None
                          else "old_" + self.old_law if getattr(self, "old_fit_", None) is not None else None)
        return self

    def outside_support(self, rows):
        """Boolean mask: admitted rows whose phi is outside the TRAIN support of the fitted law."""
        inputs = osp.peo_inputs(list(rows))
        active = np.asarray(inputs["active"], bool)
        support = getattr(self, "phi_support_", None)
        if support is None:
            return np.zeros(len(active), bool)
        phi = np.asarray(inputs["phi"], float)
        with np.errstate(invalid="ignore"):
            return active & ((phi < support[0]) | (phi > support[1]))

    def predict(self, rows, parent_log10):
        rows = list(rows)
        result = super().predict(rows, parent_log10)
        outside = self.outside_support(rows)
        if outside.any():
            result[outside] = np.asarray(parent_log10, float)[outside]
        return result

    def guard_summary(self):
        return {"version": GUARD_VERSION, "law": getattr(self, "law_kind_", None),
                "phi_support": list(self.phi_support_) if getattr(self, "phi_support_", None) else None,
                "n_law_train": getattr(self, "n_law_train_", None),
                "rule": "admitted rows with phi outside [min,max] of the law's TRAIN rows keep the parent total"}
