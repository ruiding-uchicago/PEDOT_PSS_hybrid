"""Literature-inspired greybox salt-contact hypothesis, not identified kinetics.

The signed cation-softness scale is Table 1 of DOI 10.1021/ma900327d,
which cites Marcus; it is not an eV hardness scale. No table or fit defaults
are embedded here. The paper does NOT establish these quantitative equations.
Caller must admit explicit salt contact followed by rinse, using only inputs.
Bath molarity is not film ionic strength, pH, uptake or retained ion content.

x is effective removal of the LOW-conducting volume, not measured PSS mass
loss. The volume balance assumes unchanged PEDOT volume and only low-phase
removal: no density, swelling, chain rearrangement or ion uptake is inferred.
No IO, fit, target inversion, ID/species routing or conductivity solver here.
"""
from __future__ import annotations

import math
from numbers import Real
import numpy as np

BOLTZMANN_EV_PER_K = 8.617333262e-5


def _numbers(value, name, *, unknown=False):
    obj = np.asarray(value, dtype=object)
    if obj.ndim > 1 or (obj.ndim == 1 and not obj.size):
        raise ValueError(name+" must be scalar or nonempty 1D")
    for v in obj.flat:
        if v is None and unknown:
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
            raise ValueError(name+" must contain numeric values, not booleans/strings")
    out = np.asarray(value, dtype=float)
    if np.any(np.isinf(out)) or (not unknown and not np.all(np.isfinite(out))):
        raise ValueError(name+" must be finite")
    return out


def _align(*arrays):
    lengths = {a.size for a in arrays if a.ndim == 1}
    if len(lengths) > 1:
        raise ValueError("1D inputs must have the same length; only scalars broadcast")
    shape = (next(iter(lengths)),) if lengths else ()
    return [np.broadcast_to(a, shape) for a in arrays]


def _parameter(value, name, *, lower=0., strict=False, upper=None):
    a = _numbers(value, name)
    if a.ndim or (a.item() <= lower if strict else a.item() < lower) or (upper is not None and a.item() >= upper):
        raise ValueError(name+" is outside its scalar parameter bounds")
    return a.item()


def _returned(array):
    return array.item() if array.ndim == 0 else array


def salt_contact_extent(concentration_mol_L, cation_softness, contact_temperature_K,
                        contact_seconds, *, K_ref_M, softness_slope, k_ref_per_s,
                        Ea_eV, T_ref_K, max_removed_fraction, active):
    """Return x with K=Kref exp(-slope*s), theta=C/(K+C), x=m*theta*(1-exp(-kt)).

    k=kref exp[-Ea/kB*(1/T-1/Tref)]. All shared parameters are explicit scalars:
    Kref,Tref>0; slope,kref,Ea>=0; 0<=m<1. Temperatures MUST already be Kelvin,
    time seconds, C applied-solution mol/L. active must be bool, scalar or aligned
    1D. Unknown None/NaN is allowed ONLY on inactive rows and gives exact0; it is
    not a guessed known-zero treatment. Finite malformed values are still rejected.
    """
    K = _parameter(K_ref_M, "K_ref_M", strict=True)
    slope = _parameter(softness_slope, "softness_slope")
    k = _parameter(k_ref_per_s, "k_ref_per_s")
    Ea = _parameter(Ea_eV, "Ea_eV")
    Tref = _parameter(T_ref_K, "T_ref_K", strict=True)
    maximum = _parameter(max_removed_fraction, "max_removed_fraction", upper=1.)
    mask = np.asarray(active)
    if mask.dtype.kind != "b" or mask.ndim > 1 or (mask.ndim == 1 and not mask.size):
        raise ValueError("active must be scalar or nonempty 1D bool")
    C, s, T, seconds, mask = _align(*[_numbers(v, name, unknown=True) for v, name in
        ((concentration_mol_L,"concentration_mol_L"),(cation_softness,"cation_softness"),
         (contact_temperature_K,"contact_temperature_K"),(contact_seconds,"contact_seconds"))], mask)
    for a, bad in ((C,C<0), (s,np.zeros(s.shape,dtype=bool)), (T,T<=0), (seconds,seconds<0)):
        if np.any(bad) or np.any(mask & ~np.isfinite(a)):
            raise ValueError("Active inputs must be known; C,time>=0 and T>0")
    result = np.zeros(C.shape)
    use = mask & (C>0) & (seconds>0)
    if not k or not maximum or not np.any(use):
        return _returned(result)
    c, soft, temp, time = C[use], s[use], T[use], seconds[use]
    with np.errstate(over="ignore"):
        log_K = math.log(K)-slope*soft
    theta = np.exp(-np.logaddexp(0., log_K-np.log(c)))
    shift = np.zeros(temp.shape)
    nonzero = temp != Tref
    if Ea and np.any(nonzero):
        dt = temp[nonzero]-Tref
        log_magnitude = math.log(Ea)-math.log(BOLTZMANN_EV_PER_K)+np.log(np.abs(dt))-np.log(temp[nonzero])-math.log(Tref)
        limit = math.log(np.finfo(float).max)
        magnitude = np.where(log_magnitude>limit, np.inf, np.exp(np.minimum(log_magnitude,limit)))
        shift[nonzero] = np.sign(dt)*magnitude
    log_kt = math.log(k)+np.log(time)+shift
    memory = -np.expm1(-np.exp(np.minimum(log_kt,math.log(750.))))
    result[use] = maximum*theta*memory
    return _returned(result)


def lowphase_removal_volume_balance(f, x):
    """Conditional effective-volume f/[f+(1-f)*(1-x)]; 0<=f<=1, 0<=x<1.

    This is not mass-fraction conversion or measured morphology. Scalar or aligned
    1D numeric inputs; finite values only. x0 returns f exactly, including endpoints.
    """
    f, x = _align(_numbers(f,"f"), _numbers(x,"x"))
    if np.any((f<0)|(f>1)|(x<0)|(x>=1)):
        raise ValueError("Require 0<=f<=1 and 0<=x<1")
    result = np.where(x==0, f, f/(f+(1-f)*(1-x)))
    return _returned(result)
