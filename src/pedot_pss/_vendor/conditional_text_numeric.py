"""TRAIN-only conditional text/numeric residual regression; ML, not physics.

No IO or membership inference. Caller supplies only TRAIN residual labels.
Missing/malformed numeric inputs stay unknown, never observed zeros. Text uses
the frozen field-namespaced word analyzer and finite whole-field outcome veto.
"""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
import math

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from . import recipe_text_residual as text
VERSION = "conditional-field-text-five-reported-numeric-v1"
ARMS = ("ADDITIVE", "CONDITIONAL")
NUMERIC_FIELDS = ("stock_solids_wt_percent", "dry_temperature_C", "dry_duration_min",
                  "post_treatment_temperature_C", "post_treatment_duration_min")
CONTEXTS = (("polymer_grade", "stock_solids_scope"),
            ("deposition_method", "process_sequence"),
            ("deposition_method", "process_sequence"), ("post_treatment",), ("post_treatment",))
DIAGNOSTIC_KEYS = {"n_rows", "n_features", "residual_mean", "sse", "ridge_objective",
                   "stationarity_max_abs", "intercept_stationarity_abs"}


def configuration():
    cfg = deepcopy(text.configuration(arm="word", alpha=10))
    cfg["version"] = VERSION
    cfg["text_version"] = text.VERSION
    cfg["tfidf"]["max_features"] = 6000
    cfg.update(arms=list(ARMS), numeric_fields=list(NUMERIC_FIELDS), contexts=[list(c) for c in CONTEXTS],
        numeric_transforms=["fraction=v/100;0<v<=100", "kelvin=v+273.15;v>-273.15",
            "log1p(v/60);v>=0", "kelvin=v+273.15;v>-273.15", "log1p(v/60);v>=0"],
        standardization="TRAIN-known mean/std(ddof0); std<=1e-12=>1; no-known=>mean0/scale1",
        design_order="text;five_z;five_known;CONDITIONAL adds per numeric field context_text*z then context_text*known")
    return cfg


def _finite(v):
    try:
        return not isinstance(v, (bool, np.bool_)) and isinstance(v, Real) and math.isfinite(float(v))
    except (ValueError, OverflowError):
        return False


def _project(rows, *, training=False):
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("Iterable of raw input mappings required")
    try:
        rows = list(rows)
    except TypeError as exc:
        raise ValueError("Iterable of raw input mappings required") from exc
    if training and not rows:
        raise ValueError("Nonempty TRAIN rows required")
    values = np.zeros((len(rows), 5)); known = np.zeros(values.shape, bool); audit = []
    for i, row in enumerate(rows):
        a = text.project_text(row)
        for j, key in enumerate(NUMERIC_FIELDS):
            v = row.get(key)
            if not _finite(v): continue
            v = float(v)
            if j == 0:
                if not 0. < v <= 100.: continue
                x = v/100.
            elif j in (1, 3):
                if v <= -273.15: continue
                x = v+273.15
            else:
                if v < 0.: continue
                x = math.log1p(v/60.)  # seconds/3600, without overflowing 60*v.
            if not math.isfinite(x) or (j in (0,2,4) and v>0. and x==0.):
                raise ValueError("Valid numeric transform is not representable; no clipping or unknown recoding")
            values[i,j], known[i,j] = x, True
        a.update(numeric_known=known[i].tolist(),
            numeric_transformed=[float(values[i,j]) if known[i,j] else None for j in range(5)])
        audit.append(a)
    return values, known, audit


def _vectorizer(terms=None):
    return TfidfVectorizer(analyzer=text.FieldAnalyzer("word"), dtype=np.float64,
        vocabulary=None if terms is None else {term:i for i,term in enumerate(terms)},
        **configuration()["tfidf"])


def _cross_indices(terms):
    return [[i for i, term in enumerate(terms) if term.split("\x1f", 1)[0] in fields]
            for fields in CONTEXTS]


def _dimensions(terms):
    base = len(terms)+10
    return dict(ADDITIVE=base, CONDITIONAL=base+2*sum(map(len, _cross_indices(terms))))


def _design(basis, values, known, audit):
    tr = basis["transform"]; terms = tr["terms"]
    if not terms or not audit:
        words = sparse.csr_matrix((len(audit), len(terms)), dtype=np.float64)
    else:
        vec = _vectorizer(terms); vec.idf_ = np.array(tr["idf"], dtype=np.float64)
        words = vec.transform(text._documents(audit)).tocsr()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            z = np.where(known, (values-np.array(tr["numeric_mean"]))/np.array(tr["numeric_scale"]), 0.)
        except FloatingPointError as exc:
            raise ValueError("Finite un-clipped standardized design required") from exc
    blocks = [words, sparse.csr_matrix(z), sparse.csr_matrix(known.astype(np.float64))]
    additive = sparse.hstack(blocks, format="csr")
    for j, indices in enumerate(_cross_indices(terms)):
        context = words[:,indices]
        blocks.extend((context.multiply(z[:,j,None]), context.multiply(known[:,j,None])))
    conditional = sparse.hstack(blocks, format="csr")
    designs = dict(ADDITIVE=additive, CONDITIONAL=conditional)
    if any(not np.isfinite(x.data).all() for x in designs.values()):
        raise ValueError("Finite sparse design required")
    for x in designs.values(): x.eliminate_zeros()
    return designs


def fit_design(rawrows):
    """Fit vocab/IDF and five scalers once, without labels; return basis/designs/audit."""
    values, known, audit = _project(rawrows, training=True)
    vec = _vectorizer()
    try:
        vec.fit(text._documents(audit))
        terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get); idf = vec.idf_.tolist()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc): raise
        terms, idf = [], []
    means, scales = [], []
    for j in range(5):
        v = values[known[:,j],j]
        # Stable positive-input mean/std: no finite valid point is clipped/dropped.
        magnitude = float(v.max()) if len(v) else 0.
        normalized = v/magnitude if magnitude else np.zeros(len(v))
        mean = float(normalized.mean()*magnitude) if len(v) else 0.
        std = float(normalized.std(ddof=0)*magnitude) if len(v) else 0.
        means.append(mean); scales.append(std if std>1e-12 else 1.)
    basis = dict(version=VERSION, config=configuration(), n_train=len(audit), transform=dict(
        terms=terms, idf=idf, numeric_mean=means, numeric_scale=scales,
        numeric_known_counts=known.sum(axis=0).tolist()))
    return basis, _design(basis, values, known, audit), audit


def _validate(model, *, heads=True):
    keys = {"version", "config", "n_train", "transform"} | ({"heads"} if heads else set())
    if (type(model) is not dict or model.keys()!=keys or model["version"]!=VERSION
            or not text._same(model["config"], configuration())
            or type(model["n_train"]) is not int or model["n_train"]<1):
        raise ValueError("Closed fitted state/configuration required")
    tr = model["transform"]
    if type(tr) is not dict or tr.keys()!={"terms","idf","numeric_mean","numeric_scale","numeric_known_counts"}:
        raise ValueError("Closed transform state required")
    terms = tr["terms"]
    if (type(terms) is not list or len(terms)>6000 or any(type(t) is not str or "\x1f" not in t
            or t.split("\x1f",1)[0] not in text.FIELDS or not t.split("\x1f",1)[1] for t in terms)
            or len(set(terms))!=len(terms)):
        raise ValueError("Unique field-namespaced TRAIN vocabulary required")
    for key, size in (("idf",len(terms)),("numeric_mean",5),("numeric_scale",5)):
        v = tr[key]
        if type(v) is not list or len(v)!=size or any(type(x) is not float or not _finite(x) for x in v):
            raise ValueError("Finite compact transform vectors required")
    if any(x<=0 for x in tr["idf"]+tr["numeric_scale"]): raise ValueError("Positive IDF/scales required")
    counts = tr["numeric_known_counts"]
    if (type(counts) is not list or len(counts)!=5 or any(type(x) is not int or not 0<=x<=model["n_train"] for x in counts)
            or any(c==0 and (tr["numeric_mean"][j]!=0. or tr["numeric_scale"][j]!=1.) for j,c in enumerate(counts))):
        raise ValueError("Consistent TRAIN-known counts/scales required")
    if heads:
        if type(model["heads"]) is not dict or model["heads"].keys()!=set(ARMS): raise ValueError("Both fixed heads required")
        for arm, n in _dimensions(terms).items():
            head = model["heads"][arm]
            if (type(head) is not dict or head.keys()!={"coef","intercept","diagnostics"}
                    or type(head["coef"]) is not list or len(head["coef"])!=n
                    or any(type(v) is not float or not _finite(v) for v in head["coef"])
                    or type(head["intercept"]) is not float or not _finite(head["intercept"])):
                raise ValueError("Finite aligned compact head required")
            d = head["diagnostics"]
            if (type(d) is not dict or d.keys()!=DIAGNOSTIC_KEYS or type(d["n_rows"]) is not int
                    or d["n_rows"]!=model["n_train"] or type(d["n_features"]) is not int or d["n_features"]!=n
                    or any(type(d[k]) is not float or not _finite(d[k]) or
                        (k!="residual_mean" and d[k]<0.) for k in DIAGNOSTIC_KEYS-{"n_rows","n_features"})):
                raise ValueError("Finite TRAIN diagnostics required")
    return model


def compact(model):
    """Pure JSON-safe fitted record; no estimator objects or TRAIN matrices."""
    return deepcopy(_validate(model))


def restore(record):
    """Validate/copy saved coefficients; never supervised or unsupervised fit."""
    return compact(record)


def transform(model, rawrows):
    """QUERY only: shared fitted vocabulary/scales, then both sparse designs."""
    _validate(model)
    values, known, audit = _project(rawrows)
    return _design(model, values, known, audit), audit


def _predictions(model, designs):
    result = {arm:np.asarray(designs[arm] @ np.array(model["heads"][arm]["coef"])).ravel()
        + model["heads"][arm]["intercept"] for arm in ARMS}
    if any(not np.isfinite(v).all() for v in result.values()): raise ValueError("Finite predictions required")
    return result


def predict(model, rawrows):
    return _predictions(model, transform(model, rawrows)[0])


def _labels(y, n):
    a = np.asarray(y)
    if (a.shape!=(n,) or a.dtype.kind not in "iuf" or not np.isfinite(a).all()
            or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(y,dtype=object))):
        raise ValueError("Finite aligned nonboolean TRAIN residual labels required")
    return a.astype(np.float64)


def _diagnostics(matrix, y, coef, intercept):
    residual = np.asarray(matrix @ coef).ravel()+intercept-y
    gradient = np.asarray(matrix.T @ residual).ravel()+10.*coef
    sse = float(residual @ residual)
    d = dict(n_rows=len(y), n_features=matrix.shape[1], residual_mean=float(y.mean()), sse=sse,
        ridge_objective=sse+10.*float(coef @ coef), stationarity_max_abs=float(abs(gradient).max(initial=0.)),
        intercept_stationarity_abs=float(abs(residual.sum())))
    if any(not _finite(v) for v in d.values()): raise ValueError("Finite TRAIN diagnostics required")
    return d


def fit(rawrows, residual_train):
    """One TRAIN transform fit, two Ridge fits; predictions use saved coefficients."""
    basis, designs, _ = fit_design(rawrows); y = _labels(residual_train,basis["n_train"])
    heads = {}
    for arm in ARMS:
        ridge = Ridge(**basis["config"]["ridge"]).fit(designs[arm], y)
        coef, intercept = np.asarray(ridge.coef_,dtype=np.float64), float(ridge.intercept_)
        heads[arm] = dict(coef=coef.tolist(), intercept=intercept,
            diagnostics=_diagnostics(designs[arm],y,coef,intercept))
    model = dict(basis, heads=heads); _validate(model)
    return model, _predictions(model,designs)


def replay_train(model, rawrows, residual_train):
    """Rebuild only TRAIN vocab/scalers; verify saved diagnostics without Ridge.fit."""
    _validate(model)
    if isinstance(rawrows, (str, bytes, Mapping)): raise ValueError("Iterable of TRAIN rows required")
    try:
        rawrows = list(rawrows)
    except TypeError as exc:
        raise ValueError("Iterable of TRAIN rows required") from exc
    fresh, designs, _ = fit_design(rawrows)
    if fresh["n_train"]!=model["n_train"]: raise ValueError("TRAIN count changed")
    saved, current = model["transform"], fresh["transform"]
    if saved["terms"]!=current["terms"] or saved["numeric_known_counts"]!=current["numeric_known_counts"]:
        raise ValueError("TRAIN vocabulary/known membership changed")
    error = max((float(np.max(abs(np.array(saved[k])-np.array(current[k])),initial=0.))
        for k in ("idf","numeric_mean","numeric_scale")),default=0.)
    if error>1e-12: raise ValueError("TRAIN IDF/scales changed")
    values, known, audit = _project(rawrows); old_designs = _design(model,values,known,audit)
    for arm in ARMS:
        difference = old_designs[arm]-designs[arm]
        error = max(error,float(abs(difference.data).max(initial=0.)))
    if error>1e-12: raise ValueError("TRAIN design changed")
    y = _labels(residual_train,model["n_train"]); diagnostics = {}
    for arm in ARMS:
        head=model["heads"][arm]; d=_diagnostics(designs[arm],y,np.array(head["coef"]),head["intercept"])
        if any(abs(d[k]-head["diagnostics"][k])>1e-9 for k in DIAGNOSTIC_KEYS):
            raise ValueError("Saved TRAIN diagnostics changed")
        diagnostics[arm]=d
    return dict(predictions=_predictions(model,designs),diagnostics=diagnostics,max_transform_error=error)
