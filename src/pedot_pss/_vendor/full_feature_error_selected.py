"""Pure original full-feature selected-head arithmetic; no historical runner."""
import gc
from types import SimpleNamespace
import numpy as np
import lightgbm as lgb
from scipy import sparse
from . import nonlinear_text_residual as core
def matrix(value):
    if not sparse.issparse(value) or value.ndim!=2 or value.dtype.kind not in "iuf" or not value.shape[1] or not np.isfinite(value.data).all():
        raise ValueError("Finite nonboolean CSR design required")
    value=sparse.csr_matrix(value,dtype=np.float64,copy=True);value.sum_duplicates();value.sort_indices()
    if not np.isfinite(value.data).all(): raise ValueError("Sparse aggregation overflow")
    return value

screen=SimpleNamespace(core=core,CAPACITIES=((15,32,10.),(31,8,3.),(31,16,10.),(63,8,10.)),matrix=matrix,_OWNED_FIT=lgb.LGBMRegressor.fit)
def parameters(fold,capacity):
    if type(fold) is not int or not 0<=fold<5 or type(capacity) is not int or not 0<=capacity<4:raise ValueError("Fixed original fold/capacity required")
    params=dict(screen.core.configuration()["learner"]);a,b,c=screen.CAPACITIES[capacity]
    params.update(num_leaves=a,min_child_samples=b,reg_lambda=c,random_state=20261009+fold);return params

def native(booster,fold,capacity,width):
    params=parameters(fold,capacity);aliases=dict(objective="objective",n_estimators="num_iterations",num_leaves="num_leaves",
        learning_rate="learning_rate",max_depth="max_depth",min_child_samples="min_data_in_leaf",min_child_weight="min_sum_hessian_in_leaf",
        reg_alpha="lambda_l1",reg_lambda="lambda_l2",min_split_gain="min_gain_to_split",max_bin="max_bin",min_data_in_bin="min_data_in_bin",
        subsample="bagging_fraction",subsample_freq="bagging_freq",colsample_bytree="feature_fraction",boost_from_average="boost_from_average",
        zero_as_missing="zero_as_missing",deterministic="deterministic",force_col_wise="force_col_wise",n_jobs="num_threads",verbosity="verbosity",random_state="seed")
    actual=booster._get_loaded_param();wanted={alias:params[key] for key,alias in aliases.items()};wanted["boosting"]="gbdt"
    if (any(k not in actual or (type(actual[k]) is bool)!=(type(v) is bool) or actual[k]!=v for k,v in wanted.items()) or
        booster.num_feature()!=width or not 1<=booster.num_trees()<=900 or booster.current_iteration()!=booster.num_trees() or booster.num_model_per_iteration()!=1 or
        booster.feature_name()!=[f"f{i}" for i in range(width)]):raise ValueError("Actual dynamic width/chosen capacity/native tree configuration differs")
    return actual

def fit_head(train,query,residual,fold,capacity,offset):
    train,query=screen.matrix(train),screen.matrix(query);y=np.asarray(residual)
    if (y.shape!=(train.shape[0],) or not len(y) or y.dtype.kind not in "iuf" or not np.isfinite(y).all() or
        any(isinstance(v,(bool,np.bool_)) for v in np.asarray(residual,dtype=object).flat) or query.shape[1]!=train.shape[1] or
        type(offset) is not float or offset!=float(y.mean())):raise ValueError("Finite exact TRAIN residual mean required")
    centered=y-offset
    if not np.isfinite(centered).all() or np.max(abs(centered),initial=0.)>np.finfo(np.float32).max:raise ValueError("Native TRAIN labels not representable")
    estimator=lgb.LGBMRegressor(**parameters(fold,capacity));booster=None
    try:
        screen._OWNED_FIT(estimator,train,centered,feature_name=[f"f{i}" for i in range(train.shape[1])]);booster=estimator.booster_
        actual=native(booster,fold,capacity,train.shape[1]);pt,pq=predictions(booster,train,query,offset)
        head=dict(n_train=len(y),n_features=train.shape[1],actual_trees=booster.num_trees(),effective_parameters=actual,
            model_string=booster.model_to_string(start_iteration=0,num_iteration=-1),diagnostics=screen.core._diagnostics(train,y,pt,offset))
        return head,pt,pq
    finally:
        if booster is not None:booster.free_dataset()
        del booster,estimator;gc.collect()

def predictions(booster,train,query,offset):
    result=[]
    for x in (train,query):
        pred=np.asarray(booster.predict(x,raw_score=True,start_iteration=0,num_iteration=-1,num_threads=2),float)+offset if x.shape[0] else np.empty(0)
        if pred.shape!=(x.shape[0],) or not np.isfinite(pred).all():raise ValueError("Finite aligned all-tree forward required")
        result.append(pred)
    return tuple(result)
