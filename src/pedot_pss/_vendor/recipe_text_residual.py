"""TRAIN-fitted input-text embeddings and Ridge residual correction; no IO.

Caller owns TRAIN membership and supplies only TRAIN residual labels. These are
text embeddings, not new physical quantities. Whole fields containing explicit
electrical/structural output or numeric result statements are omitted, never word-redacted.
Vetoes are finite conservative patterns, not exhaustive semantic detection.
Unknown text and scientific unit spelling/case are retained before tokenization.
"""
from collections.abc import Mapping
from numbers import Real
import json
import math
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

VERSION = "whitelisted-recipe-text-residual-v1"
FIELDS = ("polymer_grade", "stock_solids_scope", "pedot_pss_ratio_raw",
          "additive_names", "additive_amounts_raw", "dose_basis",
          "deposition_method", "substrate", "specimen_geometry",
          "process_sequence", "post_treatment", "measurement_method")
_NUMBER = r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
_OUTCOME = re.compile(
    r"conductiv(?:ity|ities)|conductance|resistivity|[σΣΩΩ]|\b(?:[kMGTmunpµμ]?ohms?|siemens)\b|"
    r"seebeck|\bpower\s+factor\b|\bcarrier\s+(?:density|concentration)\b|电导率|电阻率|"
    r"\b(?:sheet|surface)[\s_-]+resistance\b|"
    r"\bresistance\s*(?:(?:was|is)\s*)?(?:(?:of|measured(?:\s+(?:as|to\s+be))?)\s*)?"
    r"[:=]?\s*[<>≤≥~≈]?\s*" + _NUMBER + r"|"
    r"\b[munpµμk]?S\s*(?:/\s*(?:cm|m)\b|(?:[·*.]\s*)?(?:cm|m)\s*"
    r"(?:\^?\s*\{?\s*[-−–]\s*1\s*\}?|⁻¹))|"
    r"\b(?:result(?:s|ing)?|yielded|achieved|reached|measured(?:\s+value)?|"
    r"was\s+found(?:\s+to\s+be)?|结果)\s*(?:[:=]|of\b|was\b|were\b|is\b|are\b|to\s+be\b)?\s*"
    r"[<>≤≥~≈]?\s*" + _NUMBER, re.IGNORECASE)
_RESULT = (r"\s*(?:(?:is|was|were|of|at|to|by|shifted\s+(?:to|by|from))\s*|[:=]\s*)?"
           r"[<>≤≥~≈]?\s*" + _NUMBER)
_STRUCTURAL = re.compile(
    r"(?:\braman\s+(?:[^\s;,\n]+\s+){0,4}?(?:peaks?(?:\s+(?:position|shift))?|bands?|shift)|"
    r"\bXPS\s+(?:[^\s;,\n]+\s+){0,5}?(?:ratio|atomic(?:\s+(?:fraction|percent(?:age)?|content|"
    r"concentration|composition))?(?:\s+[A-Z][a-z]?)?)|"
    r"\b(?:XRD|GIWAXS|WAXS)\s+(?:[^\s;,\n]+\s+){0,4}?(?:peaks?|lattice(?:\s+parameter)?|spacing)|"
    r"\b(?:crystallinity|coherence\s+length|roughness)(?:\s+(?:RMS|Ra|Rq))?)\b" + _RESULT,
    re.IGNORECASE)


def project_text(row):
    """Read only FIELDS; lists/tuples of strings form one indivisible field."""
    if not isinstance(row, Mapping):
        raise ValueError("Raw input row must be a mapping")
    pieces, masks, reasons, fields = [], {}, {}, {}
    for key in FIELDS:
        value = row.get(key)
        if value is None or (type(value) is str and value == "") or (
                type(value) in (list, tuple) and not value):
            text, reason = "", "missing"
        elif type(value) is str:
            text, reason = value, "included"
        elif type(value) in (list, tuple) and all(type(v) is str for v in value):
            text, reason = "\n".join(value), "included"
        else:
            text, reason = "", "unsupported_type"
        if reason == "included" and not text.strip():
            reason = "missing"
        if reason == "included" and (_OUTCOME.search(text) or _STRUCTURAL.search(text)):
            reason = "outcome_text"
        masks[key], reasons[key] = reason == "included", reason
        if masks[key]:
            pieces.append(text)
            fields[key] = text
    return {"text": "\n".join(pieces), "field_text": fields,
            "field_mask": masks, "reasons": reasons}


def configuration(*, arm="char", alpha=1):
    if type(arm) is not str or arm not in ("char", "word"):
        raise ValueError("arm must be char or word")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not math.isfinite(float(alpha)) or alpha not in (1, 10)):
        raise ValueError("alpha must be 1 or 10, not bool")
    analyzer = dict(type="FieldAnalyzer", base_analyzer="char_wb" if arm == "char" else "word",
                    ngram_range=[3, 5] if arm == "char" else [1, 2], lowercase=False,
                    strip_accents=None, token_pattern=r"(?u)\b\w+\b" if arm == "word" else None)
    tfidf = dict(max_features=12000, lowercase=False, strip_accents=None,
                 min_df=1, max_df=1., binary=False, norm="l2", use_idf=True,
                 smooth_idf=True, sublinear_tf=False, stop_words=None,
                 tokenizer=None, preprocessor=None, token_pattern=None)
    return dict(version=VERSION, arm=arm, alpha=float(alpha), fields=list(FIELDS), dtype="float64",
                field_analyzer=analyzer,
                tfidf=tfidf, ridge=dict(alpha=float(alpha), fit_intercept=True,
                                      solver="lsqr", tol=1e-8, max_iter=1000))


class FieldAnalyzer:
    """Pickleable: every native sklearn n-gram is bound to its input field."""
    def __init__(self, arm):
        self.arm = arm
        self.settings = configuration(arm=arm)["field_analyzer"]
        base = dict(self.settings)
        base.pop("type")
        base["analyzer"] = base.pop("base_analyzer")
        base["ngram_range"] = tuple(base["ngram_range"])
        self.base_analyzer = TfidfVectorizer(**base).build_analyzer()

    def __call__(self, document):
        fields = json.loads(document)
        if (type(fields) is not dict or any(k not in FIELDS or type(v) is not str
                                           for k, v in fields.items())):
            raise ValueError("Closed field-text payload required")
        for key in FIELDS:
            for gram in self.base_analyzer(fields.get(key, "")):
                yield key + "\x1f" + gram


def make_vectorizer(*, arm, vocabulary=None):
    """Construct only (no fit); usable for explicit saved-vocabulary replay."""
    return TfidfVectorizer(analyzer=FieldAnalyzer(arm), dtype=np.float64,
                           vocabulary=vocabulary, **configuration(arm=arm)["tfidf"])


def _documents(audit):
    return [json.dumps(a["field_text"], ensure_ascii=False, separators=(",", ":")) for a in audit]


def _rows(rows, *, training=False):
    if isinstance(rows, (Mapping, str, bytes)):
        raise ValueError("An iterable of raw input rows is required")
    try:
        audit = [project_text(row) for row in rows]
    except TypeError as exc:
        raise ValueError("An iterable of raw input rows is required") from exc
    if training and not audit:
        raise ValueError("Nonempty TRAIN rows required")
    return audit


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _model(model):
    keys = {"version", "config", "vectorizer", "ridge", "n_train", "n_features", "intercept"}
    if type(model) is not dict or model.keys() != keys or model["version"] != VERSION:
        raise ValueError("Fitted model schema/version mismatch")
    cfg = model["config"]
    if (type(cfg) is not dict or "arm" not in cfg or "alpha" not in cfg
            or not _same(cfg, configuration(arm=cfg["arm"], alpha=cfg["alpha"]))):
        raise ValueError("Fixed fitted configuration mismatch")
    n = model["n_features"]
    if (type(n) is not int or not 0 <= n <= 12000 or type(model["n_train"]) is not int
            or model["n_train"] < 1 or type(model["intercept"]) is not float
            or not math.isfinite(model["intercept"])):
        raise ValueError("Invalid fitted dimensions/intercept")
    vec, ridge = model["vectorizer"], model["ridge"]
    if n == 0:
        if vec is not None or ridge is not None:
            raise ValueError("Empty vocabulary requires explicit constant fallback")
    elif (not isinstance(vec, TfidfVectorizer) or not isinstance(ridge, Ridge)
          or not hasattr(vec, "vocabulary_") or not hasattr(ridge, "coef_")
          or len(vec.vocabulary_) != n or vec.idf_.shape != (n,)
          or any(type(k) is not str or type(v) is not int for k, v in vec.vocabulary_.items())
          or sorted(vec.vocabulary_.values()) != list(range(n))
          or not np.isfinite(vec.idf_).all() or not (vec.idf_ > 0.).all()
          or ridge.coef_.shape != (n,) or getattr(ridge, "n_features_in_", None) != n
          or not np.isfinite(ridge.coef_).all() or not np.isfinite(ridge.intercept_)
          or float(ridge.intercept_) != model["intercept"] or vec.dtype != np.float64
          or not isinstance(vec.analyzer, FieldAnalyzer) or vec.analyzer.arm != cfg["arm"]
          or not _same(vec.analyzer.settings, cfg["field_analyzer"])
          or any(vec.get_params()[k] != v for k, v in cfg["tfidf"].items())
          or any(ridge.get_params()[k] != v for k, v in cfg["ridge"].items())):
        raise ValueError("Invalid fitted estimators/configuration")
    return model


def transform(model, queryrows):
    """QUERY transform only; audit contains whitelist field masks, no row IDs."""
    model = _model(model)
    audit = _rows(queryrows)
    if not audit or model["n_features"] == 0:
        matrix = sparse.csr_matrix((len(audit), model["n_features"]), dtype=np.float64)
    else:
        matrix = model["vectorizer"].transform(_documents(audit)).tocsr()
    return matrix, audit


def predict(model, queryrows):
    matrix, _ = transform(model, queryrows)
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    result = (np.full(matrix.shape[0], model["intercept"], dtype=np.float64)
              if model["ridge"] is None else model["ridge"].predict(matrix))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite residual prediction")
    return result


def fit(rows, residual_train, *, arm="char", alpha=1):
    """Fit TRAIN vocabulary/IDF/Ridge only; return plain model and TRAIN predictions."""
    cfg = configuration(arm=arm, alpha=alpha)
    audit = _rows(rows, training=True)
    values = np.asarray(residual_train)
    if (values.shape != (len(audit),) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
            or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(residual_train, dtype=object))):
        raise ValueError("Finite aligned nonboolean TRAIN residual vector required")
    values = values.astype(np.float64)
    vec = make_vectorizer(arm=arm)
    texts = _documents(audit)
    try:
        matrix = vec.fit_transform(texts).tocsr()
        vec.vocabulary_ = {key: int(index) for key, index in vec.vocabulary_.items()}
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        vec, matrix = None, sparse.csr_matrix((len(audit), 0), dtype=np.float64)
    ridge = Ridge(**cfg["ridge"]).fit(matrix, values) if matrix.shape[1] else None
    intercept = float(ridge.intercept_) if ridge is not None else float(np.mean(values))
    model = dict(version=VERSION, config=cfg, vectorizer=vec, ridge=ridge,
                 n_train=len(audit), n_features=matrix.shape[1], intercept=intercept)
    _model(model)
    predictions = (ridge.predict(matrix) if ridge is not None
                   else np.full(len(audit), intercept, dtype=np.float64))
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite TRAIN residual prediction")
    return model, predictions
