"""Pure original TRAIN common scaler."""
import numpy as np
def array(value,shape):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in "iuf" or not np.isfinite(a).all(): raise ValueError("Finite nonboolean aligned array required")
    if not isinstance(value,np.ndarray) and any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):
        raise ValueError("Boolean numeric input forbidden")
    return a.astype(float,copy=False)

def scale(train,query):
    """Only TRAIN determines statistics. Stable nonnegative/negative finite data."""
    X=array(train,np.shape(train))
    if X.ndim!=2 or not len(X): raise ValueError("Nonempty two-dimensional TRAIN required")
    Q=array(query,(len(query),X.shape[1]))
    maximum=np.max(abs(X),axis=0);scaled=np.divide(X,maximum,out=np.zeros_like(X),where=maximum!=0.)
    mean=scaled.mean(0)*maximum;unit_sd=scaled.std(0);sd=unit_sd*maximum
    if np.any((unit_sd>0.)&(sd==0.)): raise ValueError("Unrepresentable positive TRAIN scale")
    sd=np.where(sd==0.,1.,sd)
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        try: tr=(X-mean)/sd;qu=(Q-mean)/sd
        except FloatingPointError as exc: raise ValueError("Unrepresentable standardized design") from exc
    array(tr,tr.shape);array(qu,qu.shape)
    return tr,qu,dict(mean=mean.tolist(),scale=sd.tolist())
