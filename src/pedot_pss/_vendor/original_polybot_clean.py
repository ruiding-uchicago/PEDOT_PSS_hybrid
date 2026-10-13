"""Original selected Polybot Ridge/SVR mixture with pre-experiment inputs only.

Retains process GEM alpha=.1 and .75*Ridge(alpha=.01)+.25*SVR(C=10,
gamma=.01,epsilon=.05). Optical coverage and same-assay SD columns are removed,
not estimated from the query's outcome. The route uses experiment conditions.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .clean_inputs import canonicalize, feature_view
from .calibrated_hybrid import _prepare_rows
from .hybrid import _physics

KEYS = (
    "pb_dmso_volpct_reported", "pb_eg_volpct_reported",
    "pb_first_coating_speed_mm_s", "pb_first_coating_stage_temperature_C",
    "pb_post_speed_mm_s", "pb_post_stage_temperature_C",
    "pb_post_methanol_volpct", "pb_post_ethanol_volpct", "pb_post_water_volpct",
)


def eligible_scientific(row):
    return bool(row.get("grade_ph1000") == 1 and row.get("deposition_blade_coating") == 1
        and row.get("geometry_film") == 1 and row.get("substrate_silicon") == 1
        and all(isinstance(row.get(k), (float, int, np.number))
                and np.isfinite(row[k]) for k in KEYS))


def eligible(row):
    return eligible_scientific(feature_view(canonicalize(row)))


class OriginalPolybotClean:
    version = "original-polybot-ridge-svr-experiment-only-1012"

    @staticmethod
    def _inputs(raw):
        clean = [canonicalize(row) for row in raw]
        scientific = [feature_view(row) for row in clean]
        if not all(eligible_scientific(row) for row in scientific):
            raise ValueError("Original Polybot route requires PH1000/blade/silicon/film and complete nine controls")
        matrix = np.asarray([[row[k] for k in KEYS] for row in scientific], dtype=float).reshape(len(raw), 9)
        # Same original process-GEM preparation, using only planned controls.
        prepared, _ = _prepare_rows(scientific, .35, "polybot")
        return matrix, prepared

    def fit(self, raw, y):
        matrix, prepared = self._inputs(raw)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(raw),) or not len(raw) or not np.isfinite(y).all():
            raise ValueError("Finite matching TRAIN targets required")
        self.physics_ = _physics.GreyBoxTransport(max_nfev=150, enable_hopping=False,
                                                  random_state=20261009)
        self.physics_.fit(prepared, y)
        physical = self.physics_.predict(prepared)
        residual = y - .1*physical
        self.scaler_ = StandardScaler().fit(matrix)
        x = self.scaler_.transform(matrix)
        self.ridge_ = Ridge(alpha=.01).fit(x, residual)
        self.svr_offset_ = float(residual.mean())
        self.svr_ = SVR(C=10., gamma=.01, epsilon=.05).fit(x, residual-self.svr_offset_)
        self.n_training_examples_ = len(raw)
        return self

    def predict_values(self, raw):
        if not len(raw):
            return np.empty((0, 5))
        matrix, prepared = self._inputs(raw)
        physical = self.physics_.predict(prepared)
        x = self.scaler_.transform(matrix)
        residual = .75*self.ridge_.predict(x) + .25*(self.svr_.predict(x)+self.svr_offset_)
        return np.column_stack((.1*physical+residual, physical, .1*physical,
                                residual, np.zeros(len(raw))))

    def predict(self, raw):
        return self.predict_values(raw)[:, 0]
