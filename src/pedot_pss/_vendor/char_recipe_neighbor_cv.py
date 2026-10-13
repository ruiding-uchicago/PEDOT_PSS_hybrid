"""Pure original character-neighborhood head; no historical orchestration."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from . import conditional_text_numeric as numeric
from . import recipe_text_residual as text
VERSION="ephemeral-field-character-recipe-neighbor-v1"
ARMS=("COSINE","COSINE_SQUARED")
POWERS=dict(zip(ARMS,(1,2)))
_OWNED_FIT=Ridge.fit
def configuration():
    cfg=text.configuration(arm="char",alpha=1)
    return dict(tfidf={**cfg["tfidf"],"max_features":6000},ridge=dict(alpha=.01,fit_intercept=True,solver="cholesky"))
def _array(value, shape=None):
    a=np.asarray(value)
    if a.dtype.kind not in "iuf" or (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError("Finite aligned nonboolean numeric array required")
    if not isinstance(value,np.ndarray) and any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):
        raise ValueError("Boolean numeric input forbidden")
    return np.asarray(a,dtype=np.float64)

def _stats(values):
    if not len(values): return 0.,1.
    magnitude=float(np.max(abs(values)))
    normalized=values/magnitude if magnitude else np.zeros(len(values))
    mean=float(normalized.mean()*magnitude); sd=float(normalized.std()*magnitude)
    if not np.isfinite([mean,sd]).all(): raise ValueError("TRAIN scale is not representable")
    return mean,sd if sd>1e-12 else 1.

def _vectorizer(terms=None):
    return TfidfVectorizer(analyzer=text.FieldAnalyzer("char"), dtype=np.float64,
        vocabulary=None if terms is None else {term:i for i,term in enumerate(terms)},
        **configuration()["tfidf"])

def _validate_basis(basis):
    if (type(basis) is not dict or set(basis)!={"version","n_train","seed","transform","anchor_indices","anchor_text"}
            or basis["version"]!=VERSION or type(basis["n_train"]) is not int or basis["n_train"]<1
            or type(basis["seed"]) is not int or basis["seed"]<0): raise ValueError("Closed TRAIN basis required")
    tr=basis["transform"]
    if type(tr) is not dict or set(tr)!={"terms","idf","numeric_mean","numeric_scale","numeric_known_counts","teacher_mean","teacher_scale"}:
        raise ValueError("Closed transform required")
    terms=tr["terms"]
    if (type(terms) is not list or len(terms)>6000 or
            any(type(t) is not str or "\x1f" not in t or t.split("\x1f",1)[0] not in text.FIELDS or not t.split("\x1f",1)[1] for t in terms)
            or len(set(terms))!=len(terms)):
        raise ValueError("Unique field-bound TRAIN vocabulary required")
    for name,n in (("idf",len(terms)),("numeric_mean",5),("numeric_scale",5),("teacher_mean",4),("teacher_scale",4)):
        v=tr[name]
        if type(v) is not list or any(type(x) is not float for x in v): raise ValueError("Typed compact scales required")
        _array(v,(n,))
    if any(v<=0 for v in tr["idf"]+tr["numeric_scale"]+tr["teacher_scale"]): raise ValueError("Positive IDF/scales required")
    counts=tr["numeric_known_counts"]
    if type(counts) is not list or len(counts)!=5 or any(type(v) is not int or not 0<=v<=basis["n_train"] for v in counts):
        raise ValueError("TRAIN-known counts required")
    if any(c==0 and (tr["numeric_mean"][j]!=0. or tr["numeric_scale"][j]!=1.) for j,c in enumerate(counts)):
        raise ValueError("Unknown is not an observed zero/scaler")
    ix=basis["anchor_indices"];m=min(256,basis["n_train"])
    if (type(ix) is not list or len(ix)!=m or any(type(i) is not int for i in ix) or ix!=sorted(set(ix)) or
            ix[0]<0 or ix[-1]>=basis["n_train"]): raise ValueError("Sorted distinct TRAIN anchors required")
    a=basis["anchor_text"]
    if (not sparse.isspmatrix_csr(a) or a.shape!=(m,len(terms)) or a.dtype!=np.float64 or
            not np.isfinite(a.data).all() or np.any(a.data<0) or np.any(np.asarray(a.multiply(a).sum(axis=1))>1+1e-12)):
        raise ValueError("Finite unit-norm CSR TRAIN anchors required")
    return basis

def _design(basis, values, known, audit, teacher4):
    _validate_basis(basis); tr=basis["transform"]; n=len(audit); t=_array(teacher4,(n,4))
    if tr["terms"] and n:
        vec=_vectorizer(tr["terms"]);vec.idf_=np.asarray(tr["idf"])
        words=vec.transform(text._documents(audit)).tocsr()
    else: words=sparse.csr_matrix((n,len(tr["terms"])),dtype=np.float64)
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        try:
            z=np.where(known,(values-tr["numeric_mean"])/tr["numeric_scale"],0.)
            state=(t-tr["teacher_mean"])/tr["teacher_scale"]
            cosine=(words@basis["anchor_text"].T).toarray()
            if np.any(cosine < -1e-12) or np.any(cosine>1+1e-12): raise ValueError("Cosine bound violation")
            cosine=np.clip(cosine,0.,1.)  # canonicalize only unit-norm roundoff
            common=np.column_stack((np.ones(n),z,known.astype(float),state))
            designs={arm:np.column_stack((common,cosine**power/np.sqrt(cosine.shape[1]))) for arm,power in POWERS.items()}
        except FloatingPointError as exc: raise ValueError("Unrepresentable finite design; no clipping of covariates") from exc
    if any(not np.isfinite(x).all() for x in designs.values()): raise ValueError("Finite design required")
    return designs

def fit_basis(rawrows,teacher4,*,seed):
    """One unsupervised TRAIN fit; anchor CSR is RAM-only, never an output."""
    if type(seed) is not int or seed<0: raise ValueError("Nonnegative integer seed required")
    values,known,audit=numeric._project(rawrows,training=True);t=_array(teacher4,(len(audit),4));vec=_vectorizer()
    try:
        words=vec.fit_transform(text._documents(audit)).tocsr()
        terms=sorted(vec.vocabulary_,key=vec.vocabulary_.get);idf=vec.idf_.tolist()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc): raise
        terms,idf=[],[];words=sparse.csr_matrix((len(audit),0),dtype=np.float64)
    nm,ns=zip(*(_stats(values[known[:,j],j]) for j in range(5)))
    tm,ts=zip(*(_stats(t[:,j]) for j in range(4)))
    ix=np.sort(np.random.default_rng(seed).choice(len(audit),min(256,len(audit)),replace=False)).tolist()
    basis=dict(version=VERSION,n_train=len(audit),seed=seed,transform=dict(terms=terms,idf=idf,
        numeric_mean=list(nm),numeric_scale=list(ns),numeric_known_counts=known.sum(axis=0).tolist(),
        teacher_mean=list(tm),teacher_scale=list(ts)),anchor_indices=ix,anchor_text=words[ix].copy())
    return basis,_design(basis,values,known,audit,t),audit

def transform_basis(basis,rawrows,teacher4):
    values,known,audit=numeric._project(rawrows)
    return _design(basis,values,known,audit,teacher4),audit

def fit_head(design,residual):
    """Only captured fit on this newly owned Ridge; ancestor fits stay banned."""
    x=_array(design)
    if x.ndim!=2 or not len(x) or x.shape[1]<16: raise ValueError("Nonempty aligned TRAIN design required")
    y=_array(residual,(len(x),))
    model=Ridge(**configuration()["ridge"]);_OWNED_FIT(model,x,y)
    coef=_array(model.coef_,(x.shape[1],));intercept=float(model.intercept_);pred=x@coef+intercept;error=pred-y
    gradient=x.T@error+.01*coef;normal=float(np.max(abs(gradient),initial=0.))
    if not np.isfinite(pred).all() or abs(error.sum())>1e-7*(1+np.sum(abs(y))) or normal>1e-7*(1+float(np.max(abs(x.T@y),initial=0.))):
        raise ValueError("Finite Ridge normal equations required")
    diag=dict(n_train=len(x),n_features=x.shape[1],TRAIN_sse=float(error@error),
        TRAIN_mean_baseline_sse=float(np.sum((y-y.mean())**2)),ridge_objective=float(error@error+.01*(coef@coef)),
        normal_equation_max_abs=normal,intercept_stationarity_abs=float(abs(error.sum())))
    del model
    return dict(coef=coef.tolist(),intercept=intercept,diagnostics=diag),pred

def predict_head(head,design):
    x=_array(design)
    if (x.ndim!=2 or type(head) is not dict or set(head)!={"coef","intercept","diagnostics"}
            or type(head["intercept"]) is not float or not np.isfinite(head["intercept"])): raise ValueError("Closed Ridge head required")
    pred=x@_array(head["coef"],(x.shape[1],))+head["intercept"]
    if not np.isfinite(pred).all(): raise ValueError("Finite residual prediction required")
    return pred
