"""Chosen-only matched representation state; no supervised fit/IO/encoder load.

Projection float32 bytes are learned parameters, not raw input embeddings.
L stores no material projector; M stores no vocabulary/IDF/lexical projector.
"""
import base64
from copy import deepcopy

import numpy as np
from scipy import sparse

from . import matched_recipe_representation as core
VERSION="chosen-matched-recipe-basis-v1"


def _projection(value,p):
    core._validate_projection(value,p)
    return dict(rank=value["rank"],n_input_features=p,singular_values=value["singular_values"].tolist(),
        components_f32_le_base64=base64.b64encode(value["components"].astype("<f4",copy=False).tobytes(order="C")).decode("ascii"))


def _restore_projection(value,p):
    if type(value)is not dict or set(value)!={"rank","n_input_features","singular_values","components_f32_le_base64"} or type(value["components_f32_le_base64"])is not str:raise ValueError("Closed chosen projection required")
    expected=core.RANK*p*4
    if len(value["components_f32_le_base64"])!=4*((expected+2)//3):raise ValueError("Exact bounded projection bytes required")
    try:raw=base64.b64decode(value["components_f32_le_base64"],validate=True)
    except (ValueError,base64.binascii.Error) as exc:raise ValueError("Canonical projection encoding required") from exc
    if len(raw)!=expected or base64.b64encode(raw).decode("ascii")!=value["components_f32_le_base64"]:raise ValueError("Canonical float32 byte count required")
    result=dict(rank=value["rank"],n_input_features=value["n_input_features"],singular_values=np.asarray(value["singular_values"]),
        components=np.frombuffer(raw,dtype="<f4").reshape(core.RANK,p).astype(np.float32,copy=True))
    core._validate_projection(result,p);return result


def compact(basis,arm):
    core.validate_basis(basis)
    if arm not in core.ARMS or type(arm)is not str:raise ValueError("One chosen L/M arm required")
    chosen=basis["lexical"] if arm=="L" else basis["material"]
    result=dict(version=VERSION,core_config=core.configuration(),arm=arm,fold=basis["fold"],n_train=basis["n_train"],common=deepcopy(basis["common"]))
    if arm=="L":result["lexical"]=dict(terms=list(chosen["terms"]),idf=list(chosen["idf"]),projection=_projection(chosen["projection"],len(chosen["terms"])))
    else:result["material"]=_projection(chosen,core.N_FIELDS*core.HIDDEN)
    restore(result);return result


def fit_chosen(rows,teacher4,field_vectors,*,arm,fold):
    """Export-only TRAIN unsupervised fit, identical chosen core arithmetic."""
    if type(arm)is not str or arm not in core.ARMS:raise ValueError("One chosen L/M arm required")
    core._fold(fold);values,known,audit=core.numeric._project(rows,training=True);n=len(audit)
    if not n:raise ValueError("Nonempty chosen TRAIN required")
    states=core._array(teacher4,(n,4));nm,ns=zip(*(core._stats(values[known[:,j],j]) for j in range(5)));tm,ts=zip(*(core._stats(states[:,j]) for j in range(4)))
    common=dict(numeric_mean=list(nm),numeric_scale=list(ns),numeric_known_counts=known.sum(0).tolist(),teacher_mean=list(tm),teacher_scale=list(ts))
    result=dict(version=VERSION,core_config=core.configuration(),arm=arm,fold=fold,n_train=n,common=common)
    if arm=="L":
        vec=core._vectorizer()
        try:matrix=vec.fit_transform(core.text._documents(audit)).tocsr();terms=sorted(vec.vocabulary_,key=vec.vocabulary_.get);idf=vec.idf_.tolist()
        except ValueError as exc:
            if "empty vocabulary"not in str(exc):raise
            terms,idf=[],[];matrix=sparse.csr_matrix((n,0),dtype=np.float64)
        projection=core._fit_projection(matrix,fold);result["lexical"]=dict(terms=terms,idf=idf,projection=_projection(projection,len(terms)))
        # The immutable core returns canonical transform(), not fit_transform().
        # Rebuild that exact float path before fitting the chosen head.
        if terms:
            canonical=core._vectorizer(terms);canonical.idf_=np.asarray(idf)
            matrix=canonical.transform(core.text._documents(audit)).tocsr()
        field_known=np.array([[r["field_mask"][key] for key in core.text.FIELDS] for r in audit],bool).reshape(n,core.N_FIELDS)
    else:matrix,field_known=core._material(field_vectors,audit);projection=core._fit_projection(matrix,fold);result["material"]=_projection(projection,core.N_FIELDS*core.HIDDEN)
    restore(result)
    return result,_design(projection,matrix,values,known,field_known,states,common),audit


def restore(record):
    arm=record.get("arm") if type(record)is dict else None
    wanted={"version","core_config","arm","fold","n_train","common","lexical" if arm=="L" else "material"}
    if arm not in core.ARMS or set(record)!=wanted or record["version"]!=VERSION or not core.text._same(record["core_config"],core.configuration()) or type(record["n_train"])is not int or record["n_train"]<1:raise ValueError("Closed chosen-only basis/config required")
    core._fold(record["fold"]);common=record["common"]
    if type(common)is not dict or set(common)!={"numeric_mean","numeric_scale","numeric_known_counts","teacher_mean","teacher_scale"}:raise ValueError("Closed common26 scalers required")
    for name,n in (("numeric_mean",5),("numeric_scale",5),("teacher_mean",4),("teacher_scale",4)):core._array(common[name],(n,))
    counts=common["numeric_known_counts"]
    if type(counts)is not list or len(counts)!=5 or any(type(c)is not int or not 0<=c<=record["n_train"] for c in counts) or any(common["numeric_scale"][j]<=0. or (c==0 and (common["numeric_mean"][j]!=0. or common["numeric_scale"][j]!=1.)) for j,c in enumerate(counts)) or np.any(np.asarray(common["teacher_scale"])<=0.):raise ValueError("Positive TRAIN-only scalers required")
    result={k:deepcopy(record[k]) for k in wanted if k not in ("lexical","material")}
    if arm=="L":
        lex=record["lexical"]
        if type(lex)is not dict or set(lex)!={"terms","idf","projection"}:raise ValueError("Closed chosen lexical basis required")
        terms=lex["terms"]
        if type(terms)is not list or len(terms)>6000 or len(set(terms))!=len(terms) or any(type(t)is not str or "\x1f"not in t or t.split("\x1f",1)[0]not in core.text.FIELDS or not t.split("\x1f",1)[1] for t in terms):raise ValueError("Canonical field-bound TRAIN vocabulary required")
        idf=core._array(lex["idf"],(len(terms),))
        if np.any(idf<=0.):raise ValueError("Positive saved IDF required")
        result["lexical"]=dict(terms=list(terms),idf=idf,projection=_restore_projection(lex["projection"],len(terms)))
        projection=result["lexical"]["projection"]
    else:result["material"]=_restore_projection(record["material"],core.N_FIELDS*core.HIDDEN);projection=result["material"]
    if projection["rank"]>record["n_train"]:raise ValueError("Rank exceeds TRAIN support")
    return result


def _design(projection,matrix,values,known,field_known,states,common):
    n=len(values)
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        try:
            z=np.where(known,(values-common["numeric_mean"])/common["numeric_scale"],0.);state=(states-common["teacher_mean"])/common["teacher_scale"]
            result=np.column_stack((core._project(matrix,projection),z,known.astype(float),state,field_known.astype(float)))
        except FloatingPointError as exc:raise ValueError("Unrepresentable chosen QUERY design") from exc
    if result.shape!=(n,90) or not np.isfinite(result).all():raise ValueError("Finite chosen90-column design required")
    return result


def transform(record,rows,teacher4,field_vectors=None):
    """Only chosen fitted multiplication/scaling; no fitting or QUERY adaptation."""
    basis=restore(record);values,known,audit=core.numeric._project(rows);n=len(audit)
    states=core._array(teacher4,(n,4));common=basis["common"]
    if basis["arm"]=="L":
        lex=basis["lexical"]
        if n and lex["terms"]:
            vec=core._vectorizer(lex["terms"]);vec.idf_=lex["idf"];matrix=vec.transform(core.text._documents(audit)).tocsr()
        else:matrix=sparse.csr_matrix((n,len(lex["terms"])),dtype=np.float64)
        field_known=np.array([[r["field_mask"][key] for key in core.text.FIELDS] for r in audit],bool).reshape(n,core.N_FIELDS);projection=lex["projection"]
    else:matrix,field_known=core._material(field_vectors,audit);projection=basis["material"]
    return _design(projection,matrix,values,known,field_known,states,common),audit
