"""Matched lexical/materials-language representations; pure RAM-only ML core.

No IO, encoder/model-asset loading, membership/ID lookup or automatic promotion.
Caller supplies frozen 12x768 field vectors and TRAIN-only residual labels.
Vectors are not new physics; provenance/encoder compatibility belongs to caller.
"""
from copy import deepcopy
from importlib.metadata import version
import math

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from . import conditional_text_numeric as numeric
from . import recipe_text_residual as text
VERSION="matched-field-lexical-materials-rank64-v1"
ARMS=("L","M")
RANK,HIDDEN,N_FIELDS,N_FEATURES=64,768,len(text.FIELDS),90
STATE_COLUMNS=(0,2,3,4)
_OWNED_FIT=lgb.LGBMRegressor.fit
_VERSIONS={k:version(k) for k in ("lightgbm","scikit-learn","numpy","scipy")}


def configuration():
    cfg=text.configuration(arm="char",alpha=1)
    learner=dict(objective="regression",n_estimators=300,learning_rate=.035,
        num_leaves=15,max_depth=-1,min_child_samples=20,min_child_weight=.001,
        reg_alpha=0.,reg_lambda=3.,min_split_gain=0.,max_bin=63,min_data_in_bin=3,
        subsample=1.,subsample_freq=0,colsample_bytree=1.,boost_from_average=False,
        zero_as_missing=False,deterministic=True,force_col_wise=True,n_jobs=2,verbosity=-1)
    return dict(version=VERSION,arms=list(ARMS),fields=list(text.FIELDS),
        field_analyzer=cfg["field_analyzer"],tfidf={**cfg["tfidf"],"max_features":6000},
        material_shape=[N_FIELDS,HIDDEN],material_normalization="known nonzero field then nonzero concatenated row L2; owned float32 matrix;16-row float64 scratch",
        projection=dict(rank=RANK,algorithm="randomized",n_iter=7,n_oversamples=10,
            seed_base=20261011,relative_rank_tolerance=1e-12,dtype="float32",
            decomposition_work_dtype="float64",
            scores="stored canonical float32 components; material float32/lexical float64 multiplication; float64 outputs; zero rank padding"),
        numeric_fields=list(numeric.NUMERIC_FIELDS),numeric_config=numeric.configuration(),
        state_columns=list(STATE_COLUMNS),design_order="rank64;five_z;five_known;four_scaled_parent;12_fieldknown;no constant",
        n_features=N_FEATURES,learner=learner,seed_base=20261011,versions=dict(_VERSIONS),
        offset="explicit TRAIN residual mean; no init_model or init_score",scope=__doc__.strip())


def _fold(fold):
    if type(fold)is not int or fold not in range(5): raise ValueError("Integer original fold0..4 required")
    return fold


def _array(value,shape):
    a=np.asarray(value)
    # A typed numeric ndarray cannot conceal bools; avoid millions of boxed floats.
    has_bool=False if isinstance(value,np.ndarray) and value.dtype.kind in "iuf" else any(
        isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat)
    if a.shape!=shape or a.dtype.kind not in "iuf" or has_bool or not np.isfinite(a).all():
        raise ValueError("Finite aligned nonboolean numeric values required")
    return np.asarray(a,dtype=np.float64)


def _stats(values):
    if not len(values): return 0.,1.
    magnitude=float(np.max(abs(values))); normalized=values/magnitude if magnitude else np.zeros(len(values))
    mean=float(normalized.mean()*magnitude); sd=float(normalized.std(ddof=0)*magnitude)
    if not np.isfinite([mean,sd]).all(): raise ValueError("Unrepresentable TRAIN scaler")
    return mean,sd if sd>1e-12 else 1.


def _vectorizer(terms=None):
    return TfidfVectorizer(analyzer=text.FieldAnalyzer("char"),dtype=np.float64,
        vocabulary=None if terms is None else {term:i for i,term in enumerate(terms)},
        **configuration()["tfidf"])


def _material(vectors,audit):
    n=len(audit)
    if not isinstance(vectors,np.ndarray) or vectors.shape!=(n,N_FIELDS,HIDDEN) or vectors.dtype.kind not in "iuf":
        raise ValueError("Aligned nonboolean numeric ndarray field vectors required")
    known=np.array([[r["field_mask"][key] for key in text.FIELDS] for r in audit],dtype=bool).reshape(n,N_FIELDS)
    x=np.empty(vectors.shape,dtype=np.float32)
    for start in range(0,n,16):
        stop=min(start+16,n); chunk=np.array(vectors[start:stop],dtype=np.float64,copy=True)
        if not np.isfinite(chunk).all(): raise ValueError("Finite field vectors required")
        mask=known[start:stop]
        if np.any(chunk[~mask]!=0.): raise ValueError("Vetoed/missing field vectors must be exact zero")
        magnitude=np.max(abs(chunk),axis=2,keepdims=True)
        if np.any(mask&(magnitude[:,:,0]==0.)): raise ValueError("Known field needs nonzero content vector")
        # Bounded scratch protects finite large/tiny inputs before float32 casting.
        np.divide(chunk,magnitude,out=chunk,where=magnitude!=0.)
        norms=np.sqrt(np.sum(chunk*chunk,axis=2,keepdims=True))
        np.divide(chunk,norms,out=chunk,where=norms!=0.)
        rownorm=np.sqrt(np.sum(chunk*chunk,axis=(1,2),keepdims=True))
        np.divide(chunk,rownorm,out=chunk,where=rownorm!=0.)
        x[start:stop]=chunk
    return x.reshape(n,N_FIELDS*HIDDEN),known


def _fit_projection(matrix,fold):
    n,p=matrix.shape; k=min(RANK,n,p)
    components=np.zeros((RANK,p),dtype=np.float32); singular=np.zeros(RANK)
    if k and (np.any(matrix.data) if sparse.issparse(matrix) else np.any(matrix)):
        # One short-lived dense decomposition copy; normalized material/scoring
        # stays float32. Float32 randomized SVD itself can lose orthogonality.
        work=matrix if sparse.issparse(matrix) else np.asarray(matrix,dtype=np.float64)
        if n==1 or p==1:
            dense=work.toarray() if sparse.issparse(work) else work
            _,s,v=np.linalg.svd(dense,full_matrices=False)
        else:
            fitted=TruncatedSVD(n_components=k,algorithm="randomized",n_iter=7,n_oversamples=10,
                random_state=20261011+fold).fit(work)
            s,v=fitted.singular_values_,fitted.components_
        if not np.isfinite(s).all() or not np.isfinite(v).all(): raise ValueError("Nonfinite TRAIN projection")
        rank=int(np.sum(s>1e-12*s[0])) if s[0]>0. else 0
        # Canonicalize the stored float32 pivot: casting can create an abs tie.
        v=v[:rank].astype(np.float32,copy=True)
        for row in v:
            if row[np.argmax(abs(row))]<0.: row*=-1.
        components[:rank]=v; singular[:rank]=s[:rank]
    else: rank=0
    return dict(components=components,singular_values=singular,rank=rank,n_input_features=p)


def _project(matrix,projection):
    components=projection["components"] if matrix.dtype==np.float32 else projection["components"].astype(np.float64)
    result=np.asarray(matrix@components.T,dtype=np.float64)
    if result.shape!=(matrix.shape[0],RANK) or not np.isfinite(result).all(): raise ValueError("Finite rank64 scores required")
    return result


def _validate_projection(value,p):
    if type(value)is not dict or set(value)!={"components","singular_values","rank","n_input_features"} or type(value["rank"])is not int or not 0<=value["rank"]<=RANK or type(value["n_input_features"])is not int or value["n_input_features"]!=p:
        raise ValueError("Closed supported TRAIN projection required")
    c=value["components"];s=_array(value["singular_values"],(RANK,));r=value["rank"]
    if not isinstance(c,np.ndarray) or c.dtype!=np.float32 or c.shape!=(RANK,p) or not np.isfinite(c).all(): raise ValueError("Finite float32 fitted components required")
    if r>p: raise ValueError("Rank cannot exceed TRAIN feature dimension")
    if np.any(c[r:]!=0.) or np.any(s[r:]!=0.) or np.any(s[:r]<=0.) or np.any(np.diff(s[:r])>0.): raise ValueError("Canonical positive singular values/zero padding required")
    # Validate the stored values in float64: high-dimensional float32 GEMM can
    # accumulate >2e-6 error for an orthonormal basis. Never alter its components.
    check=c[:r].astype(np.float64)
    gram_error=float(np.max(abs(check@check.T-np.eye(r)),initial=0.))
    negative_pivots=sum(row[np.argmax(abs(row))]<0. for row in c[:r])
    diagnostic=f"max_gram_error={gram_error:.17g}; negative_pivot_count={negative_pivots}; rank={r}; features={p}"
    if gram_error>2e-6: raise ValueError("Projection orthogonality failed: "+diagnostic)
    if negative_pivots: raise ValueError("Projection stored-pivot sign failed: "+diagnostic)


def validate_basis(basis):
    if type(basis)is not dict or set(basis)!={"version","config","fold","n_train","lexical","material","common"} or basis["version"]!=VERSION or not text._same(basis["config"],configuration()) or type(basis["n_train"])is not int or basis["n_train"]<1: raise ValueError("Closed RAM basis/config required")
    _fold(basis["fold"]); lex=basis["lexical"];common=basis["common"]
    if type(lex)is not dict or set(lex)!={"terms","idf","projection"}: raise ValueError("Closed lexical state required")
    terms=lex["terms"]
    if type(terms)is not list or len(terms)>6000 or any(type(t)is not str or "\x1f" not in t or t.split("\x1f",1)[0] not in text.FIELDS or not t.split("\x1f",1)[1] for t in terms) or len(set(terms))!=len(terms): raise ValueError("TRAIN field-bound vocabulary required")
    idf=_array(lex["idf"],(len(terms),))
    if np.any(idf<=0.): raise ValueError("Positive TRAIN IDF required")
    _validate_projection(lex["projection"],len(terms)); _validate_projection(basis["material"],N_FIELDS*HIDDEN)
    if max(lex["projection"]["rank"],basis["material"]["rank"])>basis["n_train"]: raise ValueError("Rank cannot exceed TRAIN rows")
    if type(common)is not dict or set(common)!={"numeric_mean","numeric_scale","numeric_known_counts","teacher_mean","teacher_scale"}: raise ValueError("Closed common26 scalers required")
    for name,n in (("numeric_mean",5),("numeric_scale",5),("teacher_mean",4),("teacher_scale",4)): _array(common[name],(n,))
    counts=common["numeric_known_counts"]
    if type(counts)is not list or len(counts)!=5 or any(type(c)is not int or not 0<=c<=basis["n_train"] for c in counts) or any(common["numeric_scale"][j]<=0. or (c==0 and (common["numeric_mean"][j]!=0. or common["numeric_scale"][j]!=1.)) for j,c in enumerate(counts)) or np.any(np.asarray(common["teacher_scale"])<=0.): raise ValueError("Positive TRAIN-only known/state scales required")
    return basis


def _design(basis,values,known,audit,teacher4,vectors,normalized_material=None):
    validate_basis(basis); n=len(audit); states=_array(teacher4,(n,4)); lex=basis["lexical"];common=basis["common"]
    if n and lex["terms"]:
        vec=_vectorizer(lex["terms"]);vec.idf_=np.asarray(lex["idf"])
        words=vec.transform(text._documents(audit)).tocsr()
    else: words=sparse.csr_matrix((n,len(lex["terms"])),dtype=np.float64)
    material,field_known=_material(vectors,audit) if normalized_material is None else normalized_material
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        try:
            z=np.where(known,(values-common["numeric_mean"])/common["numeric_scale"],0.)
            state=(states-common["teacher_mean"])/common["teacher_scale"]
            extra=np.column_stack((z,known.astype(float),state,field_known.astype(float)))
            designs={"L":np.column_stack((_project(words,lex["projection"]),extra)),
                "M":np.column_stack((_project(material,basis["material"]),extra))}
        except FloatingPointError as exc: raise ValueError("Unrepresentable QUERY design; no clipping/unknown recode") from exc
    if any(v.shape!=(n,N_FEATURES) or not np.isfinite(v).all() for v in designs.values()): raise ValueError("Finite matched90-column designs required")
    return designs


def fit_basis(rows,teacher4,field_vectors,*,fold):
    """One eligible outerTRAIN unsupervised fit per representation, no labels."""
    _fold(fold);values,known,audit=numeric._project(rows,training=True)
    states=_array(teacher4,(len(audit),4)); material,field_known=_material(field_vectors,audit);vec=_vectorizer()
    try:
        words=vec.fit_transform(text._documents(audit)).tocsr()
        terms=sorted(vec.vocabulary_,key=vec.vocabulary_.get);idf=vec.idf_.tolist()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc): raise
        terms,idf=[],[];words=sparse.csr_matrix((len(audit),0),dtype=np.float64)
    nm,ns=zip(*(_stats(values[known[:,j],j]) for j in range(5)));tm,ts=zip(*(_stats(states[:,j]) for j in range(4)))
    basis=dict(version=VERSION,config=configuration(),fold=fold,n_train=len(audit),
        lexical=dict(terms=terms,idf=idf,projection=_fit_projection(words,fold)),material=_fit_projection(material,fold),
        common=dict(numeric_mean=list(nm),numeric_scale=list(ns),numeric_known_counts=known.sum(0).tolist(),teacher_mean=list(tm),teacher_scale=list(ts)))
    return basis,_design(basis,values,known,audit,states,field_vectors,(material,field_known)),audit


def transform_basis(basis,rows,teacher4,field_vectors):
    """Only TRAIN-fitted multiplication/scaling; no QUERY fit or adaptation."""
    values,known,audit=numeric._project(rows)
    return _design(basis,values,known,audit,teacher4,field_vectors),audit


def restore_basis(basis):
    """Validated RAM copy; no fit, serialization or embedding/model IO."""
    return deepcopy(validate_basis(basis))


def _native(booster,fold):
    params=booster._get_loaded_param()
    wanted=dict(boosting="gbdt",objective="regression",num_iterations=300,learning_rate=.035,
        num_leaves=15,max_depth=-1,min_data_in_leaf=20,min_sum_hessian_in_leaf=.001,
        lambda_l1=0.,lambda_l2=3.,min_gain_to_split=0.,max_bin=63,min_data_in_bin=3,
        bagging_fraction=1.,bagging_freq=0,feature_fraction=1.,boost_from_average=False,
        zero_as_missing=False,deterministic=True,force_col_wise=True,num_threads=2,verbosity=-1,seed=20261011+fold)
    if any(k not in params or (type(params[k])is bool and type(v)is not bool) or params[k]!=v for k,v in wanted.items()): raise ValueError("Actual native fixed-head parameters differ")
    return params


def _matrix(X,*,training=False):
    a=np.asarray(X)
    if a.ndim!=2 or (training and not len(a)): raise ValueError("Nonempty TRAIN matrix required")
    return _array(X,(len(a),N_FEATURES))


def _diagnostics(y,pred,offset):
    e=pred-y;c=y-offset
    result=dict(n_train=len(y),n_features=N_FEATURES,residual_mean=float(y.mean()),sse=float(e@e),
        mean_baseline_sse=float(c@c),prediction_min=float(pred.min()),prediction_max=float(pred.max()))
    if any(not math.isfinite(v) for v in result.values()): raise ValueError("Finite TRAIN diagnostics required")
    return result


def _booster(head):
    keys={"version","config","fold","n_train","n_features","offset","actual_trees","model_string","effective_parameters","diagnostics"}
    if type(head)is not dict or set(head)!=keys or head["version"]!=VERSION or not text._same(head["config"],configuration()) or type(head["n_train"])is not int or head["n_train"]<1 or type(head["n_features"])is not int or head["n_features"]!=N_FEATURES or type(head["offset"])is not float or not math.isfinite(head["offset"]) or type(head["actual_trees"])is not int or not 1<=head["actual_trees"]<=300 or type(head["model_string"])is not str or not head["model_string"]: raise ValueError("Closed plain fixed head required")
    _fold(head["fold"]);b=lgb.Booster(model_str=head["model_string"])
    if b.num_feature()!=N_FEATURES or b.num_trees()!=head["actual_trees"] or not text._same(_native(b,head["fold"]),head["effective_parameters"]): raise ValueError("Saved native model dimensions/parameters differ")
    d=head["diagnostics"];keys={"n_train","n_features","residual_mean","sse","mean_baseline_sse","prediction_min","prediction_max"}
    if type(d)is not dict or set(d)!=keys or not text._same(d["n_train"],head["n_train"]) or not text._same(d["n_features"],N_FEATURES) or any(type(d[k])is not float or not math.isfinite(d[k]) for k in keys-{"n_train","n_features"}) or d["residual_mean"]!=head["offset"] or d["sse"]<0. or d["mean_baseline_sse"]<0. or d["prediction_min"]>d["prediction_max"]: raise ValueError("Finite closed saved TRAIN diagnostics required")
    return b


def fit_head(X,residual_train,*,fold):
    """Fit ONLY this new estimator on synthetic/authorized TRAIN labels."""
    _fold(fold);x=_matrix(X,training=True);y=_array(residual_train,(len(x),));offset=float(y.mean())
    if not math.isfinite(offset): raise ValueError("Finite TRAIN residual mean required")
    estimator=lgb.LGBMRegressor(**configuration()["learner"],random_state=20261011+fold)
    _OWNED_FIT(estimator,x,y-offset)
    b=estimator.booster_;prediction=offset+b.predict(x,raw_score=True,num_iteration=-1,num_threads=2)
    head=dict(version=VERSION,config=configuration(),fold=fold,n_train=len(x),n_features=N_FEATURES,offset=offset,
        actual_trees=b.num_trees(),model_string=b.model_to_string(num_iteration=-1),effective_parameters=_native(b,fold),diagnostics=_diagnostics(y,prediction,offset))
    _booster(head);return head,prediction


def predict_head(head,X):
    x=_matrix(X);b=_booster(head)
    prediction=np.empty(0) if not len(x) else head["offset"]+b.predict(x,raw_score=True,num_iteration=-1,num_threads=2)
    if prediction.shape!=(len(x),) or not np.isfinite(prediction).all(): raise ValueError("Finite matched-head QUERY prediction required")
    return prediction


def restore_head(head):
    """Validate native saved state and copy; no fitting or asset/file loading."""
    _booster(head);return deepcopy(head)


def replay_train(head,X,residual_train):
    x=_matrix(X,training=True);y=_array(residual_train,(len(x),));prediction=predict_head(head,x)
    diagnostics=_diagnostics(y,prediction,head["offset"])
    if head["n_train"]!=len(x) or any(abs(diagnostics[k]-head["diagnostics"][k])>1e-9 for k in diagnostics): raise ValueError("Saved TRAIN diagnostics differ")
    return dict(predictions=prediction,diagnostics=diagnostics)
