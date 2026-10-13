"""Pure original thermal input ledger; no dataset/source streaming."""

from collections import Counter

import math

import re

VERSION = "unique-declared-film-thermal-ledger-v1"

SCOPE = "Declared film-stage setpoint pairs; exploratory unknown atmosphere/history, no measured kinetics"

MAX_STDOUT_BYTES = 16384

_OUTCOME = re.compile(r"conductiv|resistiv|resistance|\bsigma\b|σ|sheet\s*resist|\bS\s*/\s*cm\b", re.I)

_DRY = re.compile(r"\b(?:dry|dried|drying|redry|redried|redrying)\b", re.I)

_ANNEAL = re.compile(r"\b(?:re)?anneal(?:ed|ing)?\b", re.I)

_CONTACT = re.compile(r"\b(?:soak|immers|dipp|rins|wash|contact)\w*\b", re.I)

_NONFILM = re.compile(r"\b(?:dry(?:ing)?|dried|anneal(?:ed|ing)?)\s+(?:the\s+)?(?:substrate|solution|ink|bath|mixture|dispersion)\b|\bsubstrate\s+(?:drying|annealing)\b", re.I)

_MISSING = {"not_reported", "not_applicable", "unclear", "qualitative"}

ALIASES = {
    "dry_temperature_C": ("first_dry", "temperature", "C"),
    "dry_duration_min": ("first_dry", "duration", "min"),
    "dry_duration_s": ("first_dry", "duration", "s"),
    "anneal_temperature_C": ("first_anneal", "temperature", "C"),
    "anneal_duration_min": ("first_anneal", "duration", "min"),
    "anneal_duration_s": ("first_anneal", "duration", "s"),
    "pb_first_dry_stage_temperature_C": ("first_dry", "temperature", "C"),
    "pb_first_dry_duration_s": ("first_dry", "duration", "s"),
    "pb_first_anneal_temperature_C": ("first_anneal", "temperature", "C"),
    "pb_first_anneal_duration_s": ("first_anneal", "duration", "s"),
    "pb_post_dry_stage_temperature_C": ("post_dry", "temperature", "C"),
    "pb_post_dry_duration_s": ("post_dry", "duration", "s"),
    "pb_second_anneal_temperature_C": ("second_anneal", "temperature", "C"),
    "pb_second_anneal_duration_s": ("second_anneal", "duration", "s"),
}

EXTRA = {"dry_temperature": ("first_dry", "temperature"), "dry_duration": ("first_dry", "duration"),
    "anneal_temperature": ("first_anneal", "temperature"), "anneal_duration": ("first_anneal", "duration"),
    "post_dry_temperature": ("post_dry", "temperature"), "post_dry_duration": ("post_dry", "duration"),
    "second_anneal_temperature": ("second_anneal", "temperature"), "second_anneal_duration": ("second_anneal", "duration")}

POINT = {"kind": None, "value": None, "approximate": None}

RAW_SCHEMA = {**dict.fromkeys(ALIASES), **dict.fromkeys(("post_treatment", "process_sequence",
    "post_treatment_temperature_C", "post_treatment_duration_min")),
    "extra_scalar_features": [{"name": None, "value": None, "unit": None, "stage": None}],
    "experiment": {"steps": [{"kind": None, "order": None, "method": {"status": None, "value": None},
        "parameters": [{"name": None, "quantity": {"status": None, "unit": None, "value": POINT}}]}]}}

def _convert(value, unit, quantity):
    if type(value) not in (int, float) or type(unit) is not str: raise ValueError("invalid_point")
    try: number = float(value)
    except (ValueError, OverflowError): raise ValueError("invalid_point") from None
    if not math.isfinite(number): raise ValueError("nonfinite_point")
    unit = unit.strip().lower()
    if quantity == "temperature":
        if unit in {"c", "degc", "°c", "celsius"}: result = number + 273.15
        elif unit in {"k", "kelvin"}: result = number
        else: raise ValueError("unsupported_unit")
        if not math.isfinite(result) or result <= 0.: raise ValueError("invalid_temperature")
    else:
        factor = {"s": 1., "sec": 1., "second": 1., "seconds": 1., "min": 60., "minute": 60.,
            "minutes": 60., "h": 3600., "hr": 3600., "hour": 3600., "hours": 3600.}.get(unit)
        if factor is None: raise ValueError("unsupported_unit")
        result = number * factor
        if number < 0 or not math.isfinite(result) or (number > 0 and result == 0.): raise ValueError("invalid_duration")
    return result

def _claim(value, unit, quantity, source, approximate=False):
    try: return dict(value=_convert(value, unit, quantity), source=source, approximate=approximate, error=None)
    except ValueError as exc: return dict(value=None, source=source, approximate=False, error=str(exc))

def _quantity(q, name, source):
    if q is None: return []
    if type(q) is not dict: return [_claim(None, None, name, source)]
    status = q.get("status")
    if type(status) is str and status in _MISSING: return []
    point = q.get("value")
    if status != "reported" or type(point) is not dict or point.get("kind") != "point":
        return [dict(value=None, source=source, approximate=False, error="nonreported_or_nonpoint_quantity")]
    approx = point.get("approximate", False)
    if type(approx) is not bool: return [_claim(None, None, name, source)]
    return [_claim(point.get("value"), q.get("unit"), name, source, approx)]

def _merge(claims):
    bad = [c["error"] for c in claims if c["error"]]
    known = [c for c in claims if c["value"] is not None]
    value = known[0]["value"] if known else None
    if any(not math.isclose(value, c["value"], rel_tol=1e-12, abs_tol=0.) for c in known): bad.append("alias_value_conflict")
    return value, sorted(set(bad)), [c["source"] for c in claims], any(c["approximate"] for c in claims)

def _method(value):
    if type(value) is dict and value.get("status") == "reported": value = value.get("value")
    else: return ""
    return value if type(value) is str and not _OUTCOME.search(value) else ""

def _atmosphere(text):
    found = set()
    for label, pattern in (("air", r"\bair\b"),
            ("inert", r"\b(?:nitrogen|argon|helium|n2|inert(?:\s+gas)?)\b"),
            ("vacuum", r"\bvacuum\b")):
        for m in re.finditer(pattern, text, re.I):
            before=text[max(0,m.start()-30):m.start()]
            if not re.search(r"\b(?:no|not|without)\s+(?:(?:under|in|an?|any)\s+)*$",before,re.I) and not re.match(r"[- ]free\b",text[m.end():],re.I): found.add(label)
    return next(iter(found)) if len(found) == 1 else "conflict" if found else "unknown"

def _event(role, occurrence, order, tier, claims, method=""):
    temp, te, ts, ta = _merge(claims["temperature"]); dur, de, ds, da = _merge(claims["duration"])
    gas = _atmosphere(method); veto = te + de
    if gas == "conflict": veto.append("atmosphere_conflict")
    if _NONFILM.search(method): veto.append("explicit_nonfilm_thermal_event")
    return dict(kind="anneal" if "anneal" in role else "dry", role=role, occurrence=occurrence, order=order,
        temperature_K=temp, duration_s=dur, known=dict(temperature=temp is not None, duration=dur is not None),
        active=temp is not None and dur is not None and not veto, pair_tier=tier, atmosphere=gas,
        approximate=ta or da, setpoint_not_measured_film_temperature=True,
        sources=dict(temperature=ts, duration=ds, atmosphere=["same_event_method"] if gas != "unknown" else []),
        alias_duplicate_declarations=max(0,len(ts)-1)+max(0,len(ds)-1), vetoes=sorted(set(veto)))

def project(raw):
    """Closed ID-free facts; complete history means DECLARED native history only."""
    if type(raw) is not dict: raise ValueError("Raw input object required")
    vetoes=[];events=[];aliases={};excluded=0;unverified=0;reordered=False;ambiguous_post=0;unknown_extra_stage=0
    def add(role, name, claim): aliases.setdefault(role, {"temperature":[], "duration":[]})[name].append(claim)
    text={}
    for field in ("process_sequence", "post_treatment"):
        v=raw.get(field);text[field]=v if type(v) is str and not _OUTCOME.search(v) else ""
        excluded += type(v) is str and bool(_OUTCOME.search(v))
    for key,(role,name,unit) in ALIASES.items():
        if raw.get(key) is not None: add(role,name,_claim(raw[key],unit,name,key))
    extras=raw.get("extra_scalar_features")
    if extras is not None and type(extras) is not list: vetoes.append("malformed_extra_list")
    for item in extras if type(extras) is list else []:
        if type(item) is not dict or type(item.get("name")) is not str or item["name"] not in EXTRA: continue
        role,name=EXTRA[item["name"]];source="extra_scalar_features/"+item["name"]
        stage=item.get("stage")
        allowed={role,"dry" if role=="first_dry" else "anneal" if role=="first_anneal" else "post"}
        if type(stage) is str and stage.strip().lower() in {"", "unknown", "unspecified", "not_reported", "unclear"}: unknown_extra_stage+=1
        elif stage is not None and (type(stage) is not str or stage not in allowed): vetoes.append("extra_stage_conflict")
        if item.get("value") is not None: add(role,name,_claim(item["value"],item.get("unit"),name,source))
    # Ordinary post prose does not prove a second event or repair a native one.
    post=text["post_treatment"];single=bool(post) and not re.search(r";|→|->|\bthen\b|\bfollowed\b",post,re.I)
    kind="dry" if _DRY.search(post) and not _ANNEAL.search(post) else "anneal" if _ANNEAL.search(post) and not _DRY.search(post) else None
    postrole=None;direct_post=single and kind is not None and not _CONTACT.search(post) and not _NONFILM.search(post)
    if direct_post:
        first="first_"+kind;later="post_dry" if kind=="dry" else "second_anneal"
        explicit=bool(re.search(r"\bredr(?:y|ied|ying)\b|\bdry\s+again\b|\bpost[- ]dry\b" if kind=="dry" else r"\bsecond\s+anneal(?:ed|ing)?\b|\breanneal(?:ed|ing)?\b|\banneal(?:ed|ing)?\s+again\b",post,re.I))
        native_steps=raw.get("experiment",{}).get("steps",[]) if type(raw.get("experiment")) is dict else []
        has_first=first in aliases or (type(native_steps) is list and any(type(s) is dict and s.get("kind")==kind for s in native_steps))
        if explicit or later in aliases: postrole=later
        elif not has_first: postrole=first
        else: ambiguous_post=1;direct_post=False
    if direct_post:
        aliases.setdefault(postrole,{"temperature":[],"duration":[]})
        for key,name,unit in (("post_treatment_temperature_C","temperature","C"),("post_treatment_duration_min","duration","min")):
            if raw.get(key) is not None: add(postrole,name,_claim(raw[key],unit,name,key))
    if re.search(r"\bredr(?:y|ied|ying)\b|\bdry\s+again\b|\bpost[- ]dry\b",post+";"+text["process_sequence"],re.I):
        aliases.setdefault("post_dry",{"temperature":[],"duration":[]})
    if re.search(r"\bsecond\s+anneal(?:ed|ing)?\b|\breanneal(?:ed|ing)?\b|\banneal(?:ed|ing)?\s+again\b",post+";"+text["process_sequence"],re.I):
        aliases.setdefault("second_anneal",{"temperature":[],"duration":[]})
    # The old explicit continued-setpoint declaration is NOT a general coating-T fallback.
    continued="continued post-coating stage setpoint" in text["process_sequence"].lower()
    if continued:
        for item in extras if type(extras) is list else []:
            if type(item) is dict and item.get("name")=="post_coating_stage_temperature" and item.get("value") is not None:
                add("post_dry","temperature",_claim(item["value"],item.get("unit"),"temperature","extra_scalar_features/post_coating_stage_temperature"))
    native=raw.get("experiment");steps=[]
    if native is not None:
        if type(native) is not dict or type(native.get("steps")) is not list: vetoes.append("malformed_native_steps")
        else: steps=native["steps"]
    orders=[s.get("order") if type(s) is dict else None for s in steps]
    order_ok=all(type(o) is int and o>=1 for o in orders) and len(set(o for o in orders if type(o) is int))==len(orders)
    if not order_ok: vetoes.append("ambiguous_native_order")
    if order_ok:
        reordered=orders!=sorted(orders);steps=sorted(steps,key=lambda s:s["order"])
    counts=Counter();native_roles=set()
    for step in steps:
        if type(step) is not dict or type(step.get("kind")) is not str or step["kind"] not in {"dry","anneal"}: continue
        kind=step["kind"];counts[kind]+=1;i=counts[kind]
        role=("first_dry" if i==1 else "post_dry" if i==2 else "dry_"+str(i)) if kind=="dry" else ("first_anneal" if i==1 else "second_anneal" if i==2 else "anneal_"+str(i))
        native_roles.add(role);claims={"temperature":[],"duration":[]};params=step.get("parameters")
        if type(params) is not list: vetoes.append("malformed_native_parameters");params=[]
        for j,p in enumerate(params):
            if type(p) is dict and type(p.get("name")) is str and p["name"] in claims:
                name=p["name"];claims[name]+=_quantity(p.get("quantity"),name,f"native/{kind}/{i}/{name}/{j}")
        method=_method(step.get("method"));event=_event(role,i,step.get("order") if order_ok else None,"native_step_pair",claims,method)
        for name,key in (("temperature","temperature_K"),("duration","duration_s")):
            other=aliases.get(role,{}).get(name,[]);value,bad,sources,approx=_merge(other)
            event["vetoes"]+=bad;event["approximate"] |= approx
            if value is not None:
                if event[key] is None: unverified+=1  # never fill an incomplete native event
                elif not math.isclose(value,event[key],rel_tol=1e-12,abs_tol=0.): event["vetoes"].append("native_alias_conflict")
            event["alias_duplicate_declarations"]+=len(sources) if event[key] is not None else max(0,len(sources)-1)
            event["sources"][name]+=sources
        event["vetoes"]=sorted(set(event["vetoes"]));event["active"] &= not event["vetoes"]
        events.append(event)
    for role,claims in aliases.items():
        if role in native_roles: continue
        occurrence=1 if role.startswith("first_") else 2
        method=post if direct_post and role==postrole else ""
        event=_event(role,occurrence,None,"schema_pair_assumed",claims,method)
        if role=="first_dry" and _NONFILM.search(text["process_sequence"]): event["vetoes"].append("explicit_nonfilm_thermal_event");event["active"]=False
        events.append(event)
    events.sort(key=lambda e:(e["order"] is None,e["order"] or 0,("first_dry","first_anneal","post_dry","second_anneal").index(e["role"]) if e["role"] in {"first_dry","first_anneal","post_dry","second_anneal"} else 4))
    for e in events: vetoes+=e["vetoes"]
    incomplete=sum(not all(e["known"].values()) for e in events)
    complete=bool(events) and bool(steps) and order_ok and not incomplete and not vetoes and not ambiguous_post and all(e["pair_tier"]=="native_step_pair" for e in events)
    flags=dict(incomplete_events=incomplete,alias_duplicate_declarations=sum(e["alias_duplicate_declarations"] for e in events),
        native_alias_unverifiable=unverified,unknown_atmosphere_events=sum(e["atmosphere"]=="unknown" for e in events),
        native_input_order_reordered=reordered,order_assumed=any(e["order"] is None for e in events),
        incomplete_processing_history=not complete,outcome_fields_excluded=excluded,
        ambiguous_post_thermal_event=ambiguous_post,unknown_extra_stage=unknown_extra_stage)
    return dict(version=VERSION,active=any(e["active"] for e in events) and not vetoes,events=events,flags=flags,
        vetoes=sorted(set(vetoes)),history_complete=complete,scope=SCOPE)
