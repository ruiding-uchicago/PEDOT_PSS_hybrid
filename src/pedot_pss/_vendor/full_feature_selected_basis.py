"""Chosen-view-only plain TRAIN basis; no IO, labels, fitting or physical law.

Teacher4 is caller-supplied privileged NONNESTED context, not intrinsic physics.
C0 stores no word/source state; C1 no source state. Unknown QUERY keys never
extend vocabularies or their TRAIN-observed known flags. C2 alone uses source
context, with unknown sources mapped to zero. Raw fields retain frozen vetoes.
"""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from . import full_feature_error_views as views
VERSION = "chosen-full-feature-plain-basis-v1"
_COMMON = {"version", "view", "schema", "canonical_keys", "numeric_keys",
           "category_pairs", "canonical_vectorizer"}
_REASONS = {"forbidden_family", "excluded_extra_family", "schema_type",
            "unsupported_unit_slug", "outside_fixed_builder"}


def _strings(values):
    if (type(values) is not list or any(type(v) is not str or not v for v in values)
            or values != sorted(set(values))):
        raise ValueError("Sorted unique nonempty string list required")
    return values


def _schema(schema):
    if (type(schema) is not dict or schema.keys() !=
            {"version", "numeric", "categorical", "excluded", "inventory_counts"}
            or schema["version"] != views.VERSION):
        raise ValueError("Closed canonical schema required")
    nums, cats = _strings(schema["numeric"]), _strings(schema["categorical"])
    checked = views.make_schema(dict(numeric_features=nums, categorical_features=cats))
    if checked["excluded"] or checked["numeric"] != nums or checked["categorical"] != cats:
        raise ValueError("Safe canonical schema names/types required")
    excluded, counts = schema["excluded"], schema["inventory_counts"]
    if (type(excluded) is not dict or any(type(k) is not str or not k or type(v) is not str
            or v not in _REASONS or k in nums or k in cats for k,v in excluded.items())
            or type(counts) is not dict or counts.keys() != {"numeric", "categorical", "union"}
            or any(type(v) is not int or v < 0 for v in counts.values())
            or counts["numeric"] < len(nums) or counts["categorical"] < len(cats)
            or counts["union"] != counts["numeric"]+counts["categorical"]
            or counts["union"] != len(nums)+len(cats)+len(excluded)):
        raise ValueError("Consistent canonical schema inventory metadata required")
    return schema


def _vectorizer(record, expected):
    if type(record) is not dict or record.keys() != {"feature_names", "vocabulary"}:
        raise ValueError("Closed DictVectorizer state required")
    names, vocabulary = record["feature_names"], record["vocabulary"]
    if (type(names) is not list or any(type(v) is not str or not v for v in names)
            or len(set(names)) != len(names) or names != sorted(expected)
            or len(set(expected)) != len(expected) or type(vocabulary) is not dict
            or any(type(k) is not str or type(v) is not int for k,v in vocabulary.items())
            or vocabulary != {name:i for i,name in enumerate(names)}):
        raise ValueError("Exact unique names and contiguous DictVectorizer indices required")
    vec = DictVectorizer(sparse=True, sort=True, dtype=np.float64)
    vec.feature_names_, vec.vocabulary_ = list(names), dict(vocabulary)
    return vec


def _saved_vectorizer(vec):
    if (not isinstance(vec, DictVectorizer) or vec.separator != "=" or not vec.sparse
            or not vec.sort or np.dtype(vec.dtype) != np.dtype(np.float64)):
        raise ValueError("Frozen sorted sparse float64 DictVectorizer required")
    try:
        return dict(feature_names=list(vec.feature_names_), vocabulary=dict(vec.vocabulary_))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Fitted DictVectorizer metadata required") from exc


def restore(record):
    """Validate/copy a compact record and reconstruct only its fitted encoders."""
    if (type(record) is not dict or type(record.get("version")) is not str
            or record["version"] != VERSION or type(record.get("view")) is not str
            or record["view"] not in views.VIEWS):
        raise ValueError("Chosen-view basis record required")
    view = record["view"]
    keys = _COMMON | ({"text_basis"} if view != "C0" else set())
    keys |= {"sources", "source_vectorizer"} if view == "C2" else set()
    if record.keys() != keys: raise ValueError("Closed chosen-view-only record required")
    schema = _schema(record["schema"])
    canonical, nums = _strings(record["canonical_keys"]), _strings(record["numeric_keys"])
    if (not set(canonical) <= set(schema["numeric"]+schema["categorical"])
            or nums != sorted(set(canonical)&set(schema["numeric"]))):
        raise ValueError("Consistent TRAIN-observed canonical key types required")
    pairs = record["category_pairs"]
    if (type(pairs) is not list or any(type(p) is not list or len(p) != 2
            or any(type(v) is not str or not v for v in p) for p in pairs)
            or pairs != [list(p) for p in sorted({tuple(p) for p in pairs})]
            or {p[0] for p in pairs} != set(canonical)-set(nums)):
        raise ValueError("Sorted unique TRAIN category pairs required")
    for key, value in pairs:
        match = views._EXTRA.fullmatch(key)
        if match:
            if (key != f"extra_{match.group(1)}_unit" or value != views._unit(value)
                    or value not in views._units(match.group(1)) | views._UNKNOWN_UNITS):
                raise ValueError("Closed normalized extra-unit category required")
        else:
            views._category(key, value)
    names = nums+["known::"+k for k in nums]+[k+"="+v for k,v in pairs]
    basis = deepcopy(record)
    basis["canonical_vectorizer"] = _vectorizer(record["canonical_vectorizer"], names)
    if view != "C0":
        views.core.numeric._validate(record["text_basis"], heads=False)
    if view == "C2":
        sources = _strings(record["sources"])
        if not sources or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}",s) is None for s in sources):
            raise ValueError("Nonempty explicit TRAIN source vocabulary required")
        basis["source_vectorizer"] = _vectorizer(record["source_vectorizer"],
            ["paper_context="+s for s in sources])
    return basis


def compact(basis, view):
    """Extract one view from frozen views.fit_views RAM state, without fitting."""
    if type(basis) is not dict or basis.get("version") != views.VERSION or view not in views.VIEWS:
        raise ValueError("Fitted original views basis and chosen view required")
    try:
        record = {k:deepcopy(basis[k]) for k in
            ("schema", "canonical_keys", "numeric_keys", "category_pairs")}
        record.update(version=VERSION, view=view,
            category_pairs=[list(p) for p in basis["category_pairs"]],
            canonical_vectorizer=_saved_vectorizer(basis["canonical_vectorizer"]))
        if view != "C0": record["text_basis"] = deepcopy(basis["text_basis"])
        if view == "C2":
            record.update(sources=deepcopy(basis["sources"]),
                source_vectorizer=_saved_vectorizer(basis["source_vectorizer"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("Complete chosen TRAIN basis required") from exc
    restore(record)
    return record


def transform(record, canonical, raw, teacher4, sources):
    """Return (chosen CSR, count audit), with no fitting or QUERY label access."""
    basis = restore(record)
    try:
        canonical, raw, teacher, sources = views._inputs(canonical, raw, teacher4, sources)
        projected, counts = views._project(canonical, basis["schema"])
    except (TypeError, OverflowError) as exc:
        raise ValueError("Aligned valid input-only rows required") from exc
    observed, nums = set(basis["canonical_keys"]), basis["numeric_keys"]
    pairs = {tuple(p) for p in basis["category_pairs"]}
    rows = []
    for row in projected:
        counts["query_only_keys"] += len(set(row)-observed)
        counts["unseen_categories"] += sum(isinstance(v,str) and (k,v) not in pairs for k,v in row.items())
        out = {k:v for k,v in row.items() if k in observed}
        out.update({"known::"+k:float(k in row) for k in nums})
        rows.append(out)
    vec = basis["canonical_vectorizer"]
    encoded = vec.transform(rows) if rows else sparse.csr_matrix((0,len(vec.feature_names_)))
    blocks = [encoded, sparse.csr_matrix(teacher)]
    word_width = source_width = 0
    if basis["view"] != "C0":
        words, _ = views.core.transform_basis(basis["text_basis"], raw)
        word_width = words.shape[1]; blocks.append(words)
    if basis["view"] == "C2":
        source_vec = basis["source_vectorizer"]
        source = (source_vec.transform([{"paper_context":s} for s in sources]) if rows
                  else sparse.csr_matrix((0,len(source_vec.feature_names_))))
        source_width = source.shape[1]; blocks.append(source)
        counts["unseen_sources"] = sum(s not in basis["sources"] for s in sources)
    matrix = sparse.hstack(blocks, format="csr")
    if not np.isfinite(matrix.data).all(): raise ValueError("Finite chosen sparse design required")
    audit = dict(view=basis["view"], n_rows=len(rows), n_features=matrix.shape[1],
        counts=dict(counts), canonical_encoded_width=encoded.shape[1], teacher_columns=4,
        word_numeric_width=word_width, source_width=source_width,
        observed_key_types=dict(numeric=len(nums),categorical=len(observed)-len(nums)),
        candidate_schema_counts={k:len(basis["schema"][k]) for k in ("numeric","categorical")},
        schema_exclusion_counts=dict(Counter(basis["schema"]["excluded"].values())))
    return matrix, audit
