"""Material descriptor tables for the opened components C (polymer grade) and D (substrate).

Every value is a nominal, literature/datasheet-level number that is known BEFORE an experiment is run
(no measured conductivity, no film characterisation of the row). Sources and assumptions:

C (PEDOT:PSS grade), Heraeus Clevios / Agfa Orgacon / Sigma-Aldrich product datasheets as commonly quoted:
  * PH1000: solids 1.0-1.3 wt% (1.15), PEDOT:PSS 1:2.5 w/w, high-conductivity grade (>=850 S/cm with 5% DMSO).
  * PH500 / PH510: solids 1.0-1.4 wt% (1.2), 1:2.5, high-conductivity grade (~500 S/cm with 5% DMSO).
  * Clevios P (= Baytron P, Bayer/H.C. Starck name before 2008): solids 1.2-1.4 wt% (1.3), 1:2.5, standard grade
    (~1 S/cm as supplied).
  * AI4083 (P VP AI 4083, Baytron 4083): solids 1.3-1.7 wt% (1.5), 1:6, hole-injection grade (500-5000 Ohm cm).
  * HTL Solar (Ossila M124): hole-transport formulation with wetting agent; solids ~1.0-1.2 wt% (assumed 1.1),
    ratio not published (NaN), standard conductivity class (assumed).
  * Orgacon ICP 1050 (Agfa): high-conductivity aqueous grade, solids ~1.2 wt% (approx.), ratio not published.
  * Orgacon Dry / re-dispersible pellets: dry PEDOT:PSS, user re-disperses (solids NaN), high-conductivity
    class assumed (Agfa markets it as conductive grade).
  * Screen-printing pastes (Orgacon S-203, EL-P-5015, Clevios S V3, Aldrich screen-printing type): paste/binder
    formulations, standard class; solids not used (NaN).
  * Clevios CPP 105 D: antistatic coating formulation with binder/solvent, standard class.
  * HIL grades (HIL-E100, HIL-1005): hole-injection, low class.
  * Aldrich 'conductive grade' (483095-type, 1.3 wt%: 0.5 wt% PEDOT + 0.8 wt% PSS -> PSS:PEDOT 1.6), standard;
    Aldrich 'high-conductivity, surfactant-free' (739332-type, 1.1 wt%), high class.
  * In-house syntheses: reported PSS molar mass parsed from the grade text (kDa; Mn or Mw as reported).
  * Conductivity class: 0 = hole-injection/low, 1 = standard, 2 = high-conductivity grade.
D (substrate), typical handbook values for cleaned surfaces (assumptions: as-cleaned, room temperature):
  water contact angle (deg), surface energy (mJ/m2), thermal conductivity (W/m K), electrically conductive (0/1),
  flexible (0/1), porous (0/1), polymer (0/1), inorganic (0/1), transparent (0/1), max processing T (C).
  glass 25/70/1.0; silicon (native oxide) 35/60/150; quartz 30/65/1.4; ITO 30/60/(glass-borne, 1.0, conductive);
  FTO 35/55/1.0 conductive; gold 70/45/318 conductive; Cu foil 80/45/400 conductive; PET 75/44/0.2 Tg~75;
  PEN 72/43/0.15; polyimide 70/46/0.12; PDMS 108/21/0.17; PP 102/30/0.2; PVDF 85/32/0.2; PTFE 112/19/0.25;
  polystyrene 88/40/0.15; polycarbonate 82/42/0.2; SEBS 98/30/0.17; PCL 78/43/0.2; paper 50/45/0.05 porous;
  cotton textile 30/48/0.05 porous; porous alumina (Anodisc) 20/60/1.5; alumina ceramic 30/55/25;
  APTES-silica 60/45/1.4; organic semiconductor blend 100/27/0.2; silicone gap filler 100/22/3.0.
  Free-standing / collected / lifted-off films: no substrate during measurement -> descriptors missing,
  `free_standing` = 1. A coagulation bath is a liquid (IPA: surface tension 23 mN/m, k 0.14).
These numbers are deliberately coarse; the model sees the same nominal values for every row with that material.

Checked against public sources on 2026-10-14 (corrections applied): PH 510 solids 1.5-1.9 wt% (Heraeus datasheet,
"high solids version of Clevios P"); HTL Solar 1:2.5, 1.0-1.3 wt% (Ossila M124); ICP 1050 = Sigma 739332,
1.1 wt%; CPP 105 D ~1.2 wt% solids (Clevios coating guide); HIL-1005 is a high-conductivity grade (1.0 wt%,
110 Ohm/sq at 90% VLT) and HIL-E100 at least standard; screen pastes (S V3, EL-P 5015) reach PH500-like sheet
resistance -> class 2; PDMS k 0.27 W/m K (Sylgard 184 TDS); paper k ~0.1 W/m K through-plane. PH500 datasheet:
solids 1.0-1.3 wt%, ~300 S/cm with DMSO. Confirmed: PH1000, Clevios P, AI 4083, Sigma 483095/739332, Orgacon DRY,
polymer-film contact angles and surface energies (Accu Dyne), Kapton, PEN/PET temperature limits.

Vendored from the m-open study (unchanged).
"""
from __future__ import annotations

import math
import re

NAN = float("nan")

# ----------------------------------------------------------------------------- C: polymer grade
# id: (pss_per_pedot_ww, solids_wt_pct, cond_class, is_paste, is_dry, is_in_house, counterion_pss, is_pedot,
#      formulated_with_additive)
GRADES = {
    "ph1000":               (2.5, 1.15, 2, 0, 0, 0, 1, 1, 0),
    "ph500":                (2.5, 1.2, 2, 0, 0, 0, 1, 1, 0),
    "ph510":                (2.5, 1.7, 2, 0, 0, 0, 1, 1, 0),
    "ph_unspecified":       (2.5, 1.2, 2, 0, 0, 0, 1, 1, 0),
    "clevios_p":            (2.5, 1.3, 1, 0, 0, 0, 1, 1, 0),
    "ai4083":               (6.0, 1.5, 0, 0, 0, 0, 1, 1, 0),
    "htl_solar":            (2.5, 1.15, 1, 0, 0, 0, 1, 1, 1),
    "icp1050":              (NAN, 1.1, 2, 0, 0, 0, 1, 1, 0),
    "redispersible_dry":    (NAN, NAN, 2, 0, 1, 0, 1, 1, 0),
    "screen_paste":         (NAN, NAN, 2, 1, 0, 0, 1, 1, 1),
    "cpp105d":              (NAN, 1.2, 1, 0, 0, 0, 1, 1, 1),
    "hil":                  (NAN, NAN, 0, 0, 0, 0, 1, 1, 0),
    "hil_1005":             (NAN, 1.0, 2, 0, 0, 0, 1, 1, 0),
    "hil_e100":             (NAN, NAN, 1, 0, 0, 0, 1, 1, 0),
    "aldrich_conductive":   (1.6, 1.3, 1, 0, 0, 0, 1, 1, 0),
    "high_cond_sf":         (NAN, 1.1, 2, 0, 0, 0, 1, 1, 0),
    "pt100":                (NAN, NAN, NAN, 0, 0, 0, 1, 1, 0),
    "commercial_unspecified": (NAN, NAN, NAN, 0, 0, 0, 1, 1, 0),
    "in_house_pss":         (NAN, NAN, NAN, 0, 0, 1, 1, 1, 0),
    "pedot_tos_otf":        (NAN, NAN, NAN, 0, 0, 1, 0, 1, 0),
    "pedot_self_doped":     (NAN, NAN, NAN, 0, 0, 1, 0, 1, 0),
    "non_pedot":            (NAN, NAN, NAN, 0, 0, 0, 0, 0, 0),
}
GRADE_FIELDS = ("pss_per_pedot", "solids_wt_pct", "cond_class", "is_paste", "is_dry_redispersible",
                "is_in_house", "counterion_pss", "is_pedot", "formulated_with_additive")

_GRADE_RULES = (   # (regex on lower-case grade text, id); every matching rule contributes (multi-grade rows)
    (r"ph\s*-?\s*1000|pss1000\b(?!\s*\()", "ph1000"),
    (r"ph\s*-?\s*500\b|ph500", "ph500"),
    (r"ph\s*-?\s*510", "ph510"),
    (r"^ph$", "ph_unspecified"),
    (r"baytron[\s-]*p\b(?!\s*synthesis)|clevi?ous\s*p\b(?!h)|^p$|\bclevios p\b", "clevios_p"),
    (r"ai\s*4083|al\s*4083|baytron\s*4083", "ai4083"),
    (r"htl\s*solar", "htl_solar"),
    (r"icp[\s-]*1050", "icp1050"),
    (r"orgacon\s*dry|re-?dispersible|redispersible|nanoparticle powder|pedot:pss particles", "redispersible_dry"),
    (r"s-?203|el-?p-?5015|sv3|screen[- ]print", "screen_paste"),
    (r"cpp\s*105", "cpp105d"),
    (r"hil-?1005", "hil_1005"),
    (r"hil-?e100", "hil_e100"),
    (r"\bhil\b(?!-?(?:1005|e100))", "hil"),
    (r"aldrich conductive grade|conductive grade; 1\.3 wt", "aldrich_conductive"),
    (r"high[- ]conductivity|high conductivity grade|\bhcg\b", "high_cond_sf"),
    (r"pt-?100", "pt100"),
    (r"in-house|in house|laboratory-synthesized|synthesized|synthesised|prepared by|atrp|raft|"
     r"polymeri[sz]ed pedot:pss|baytron-p synthesis|low-na", "in_house_pss"),
    (r"pedot[:/ -]*o?tf\b|pedot[:/ -]*tos|tosylate", "pedot_tos_otf"),
    (r"self-doped", "pedot_self_doped"),
    (r"cnf|p3ht|^ito$|sebs|^poly\(vinyl alcohol\)$|pedot-free|no pedot:pss matrix", "non_pedot"),
    (r"commercial|heraeus|h\.c|starck|agfa|philips|ossila|aldrich|kaiwei|supplier-provided|"
     r"conductive solution|aqueous dispersion|dispersion; concentration|conductive grade|aqueous stock",
     "commercial_unspecified"),
)
_SOLIDS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*wt\.?\s*%")
_PSS_MASS = re.compile(r"(?:pss[a-z]*[^;]*?(?:mw|mn|m\.w\.)\s*[=~]?\s*|pss-)(\d[\d,\.]*)\s*(k|kda|kg\s*mol|g\s*/?\s*mol|g mol)?",
                       re.I)


def grade_ids(text):
    """Canonical grade ids for a polymer_grade string (open-space CV grouping and descriptor lookup)."""
    s = (text or "").strip().lower()
    if not s or s in {"pedot:pss (unspecified grade)", "pedot", "pedot:pss"}:
        return []
    found = []
    for pattern, gid in _GRADE_RULES:
        if re.search(pattern, s) and gid not in found:
            found.append(gid)
    # specific commercial ids make the generic commercial / in-house tags redundant
    specific = [g for g in found if g not in ("commercial_unspecified", "in_house_pss")]
    if specific and "commercial_unspecified" in found:
        found.remove("commercial_unspecified")
    if "in_house_pss" in found and any(g in found for g in ("pedot_tos_otf", "pedot_self_doped", "non_pedot")):
        found.remove("in_house_pss")
    if "ph1000" in found and "clevios_p" in found and "baytron" not in s and "clevios p" not in s:
        found.remove("clevios_p")
    if not found and pss_mass_kda(s) == pss_mass_kda(s):          # reported PSS molar mass -> in-house synthesis
        found = ["in_house_pss"]
    return found or ["other_reported"]


def pss_mass_kda(text):
    s = (text or "").lower()
    if "pss" not in s:
        return NAN
    m = _PSS_MASS.search(s)
    if not m:
        return NAN
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return NAN
    unit = (m.group(2) or "").lower()
    if unit.startswith("g"):
        value /= 1000.0
    elif not unit and value > 5000:
        value /= 1000.0
    return value if 1 <= value <= 5000 else NAN


def grade_features(text, prefix="openC_"):
    ids = grade_ids(text)
    out = {prefix + "n_ids": float(len(ids)), prefix + "known": float(bool(ids) and ids != ["other_reported"])}
    rows = [GRADES[g] for g in ids if g in GRADES]
    for j, name in enumerate(GRADE_FIELDS):
        vals = [r[j] for r in rows if not (isinstance(r[j], float) and math.isnan(r[j]))]
        if vals:
            out[prefix + name] = float(sum(vals) / len(vals))
    mass = pss_mass_kda(text)
    if mass == mass:
        out[prefix + "pss_mass_kDa_log10"] = math.log10(mass)
    m = _SOLIDS.search((text or "").lower())
    if m and "pss mw" not in (text or "").lower():                  # explicitly stated dispersion solids
        lo = float(m.group(1)); hi = float(m.group(2)) if m.group(2) else lo
        if 0.1 <= lo <= 10:
            out[prefix + "solids_wt_pct"] = (lo + hi) / 2
    return out


# ----------------------------------------------------------------------------- D: substrate
# id: (wca_deg, surface_energy, k_W_mK, conductive, flexible, porous, polymer, inorganic, transparent, max_T_C)
SUBSTRATES = {
    "glass":          (25, 70, 1.0, 0, 0, 0, 0, 1, 1, 500),
    "silicon":        (35, 60, 150, 0, 0, 0, 0, 1, 0, 1000),
    "quartz":         (30, 65, 1.4, 0, 0, 0, 0, 1, 1, 1000),
    "quartz_crystal": (30, 65, 8.0, 0, 0, 0, 0, 1, 1, 1000),
    "ito":            (30, 60, 1.0, 1, 0, 0, 0, 1, 1, 300),
    "fto":            (35, 55, 1.0, 1, 0, 0, 0, 1, 1, 500),
    "gold":           (70, 45, 318, 1, 0, 0, 0, 1, 0, 500),
    "copper_foil":    (80, 45, 400, 1, 1, 0, 0, 1, 0, 300),
    "pet":            (75, 44, 0.2, 0, 1, 0, 1, 0, 1, 120),
    "pen":            (72, 43, 0.15, 0, 1, 0, 1, 0, 1, 180),
    "polyimide":      (70, 46, 0.12, 0, 1, 0, 1, 0, 0, 350),
    "pdms":           (108, 21, 0.27, 0, 1, 0, 1, 0, 1, 200),
    "polypropylene":  (102, 30, 0.2, 0, 1, 0, 1, 0, 0, 130),
    "pvdf":           (85, 32, 0.2, 0, 1, 0, 1, 0, 0, 150),
    "ptfe":           (112, 19, 0.25, 0, 0, 0, 1, 0, 0, 260),
    "polystyrene":    (88, 40, 0.15, 0, 0, 0, 1, 0, 1, 90),
    "polycarbonate":  (82, 42, 0.2, 0, 0, 0, 1, 0, 1, 130),
    "sebs":           (98, 30, 0.17, 0, 1, 0, 1, 0, 1, 100),
    "pcl":            (78, 43, 0.2, 0, 1, 0, 1, 0, 0, 55),
    "paper":          (50, 45, 0.1, 0, 1, 1, 1, 0, 0, 150),
    "textile":        (30, 48, 0.05, 0, 1, 1, 1, 0, 0, 150),
    "alumina_porous": (20, 60, 1.5, 0, 0, 1, 0, 1, 0, 600),
    "alumina":        (30, 55, 25, 0, 0, 0, 0, 1, 0, 1000),
    "aptes_silica":   (60, 45, 1.4, 0, 0, 0, 0, 1, 1, 300),
    "organic_semiconductor": (100, 27, 0.2, 0, 0, 0, 1, 0, 0, 150),
    "silicone_gap_filler": (100, 22, 3.0, 0, 1, 0, 1, 0, 0, 200),
    "polymer_membrane": (NAN, NAN, 0.2, 0, 1, 1, 1, 0, 0, NAN),
    "hydrogel_template": (40, 50, 0.5, 0, 1, 1, 1, 0, 0, 100),
    "liquid_bath_ipa": (NAN, 23, 0.14, 0, 0, 0, 0, 0, 1, NAN),
    "unspecified":    (NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN),
}
SUBSTRATE_FIELDS = ("wca_deg", "surface_energy", "log10_k_W_mK", "conductive", "flexible", "porous", "polymer",
                    "inorganic", "transparent", "max_T_C")
_SUB_RULES = (
    (r"z-cut quartz", "quartz_crystal"), (r"quartz", "quartz"), (r"glass", "glass"),
    (r"aptes", "aptes_silica"), (r"silicon|\bsi\b|wafer", "silicon"),
    (r"\bito\b", "ito"), (r"\bfto\b", "fto"), (r"gold|\bau\b", "gold"), (r"cu foil|copper", "copper_foil"),
    (r"\bpet\b", "pet"), (r"\bpen\b", "pen"), (r"polyimide|kapton", "polyimide"), (r"pdms", "pdms"),
    (r"pvdf", "pvdf"), (r"ptfe", "ptfe"), (r"polystyrene|petri dish", "polystyrene"),
    (r"polycarbonate", "polycarbonate"), (r"sebs", "sebs"), (r"pcl", "pcl"),
    (r"paper", "paper"), (r"textile|fabric|cotton", "textile"), (r"anodisc", "alumina_porous"),
    (r"alumina", "alumina"), (r"ptb7|active layer", "organic_semiconductor"),
    (r"gap-filler", "silicone_gap_filler"), (r"filament membrane", "polymer_membrane"),
    (r"polypropylene|\(pp\)", "polypropylene"), (r"membrane", "polymer_membrane"),
    (r"pva hydrogel", "hydrogel_template"), (r"coagulation bath", "liquid_bath_ipa"),
    (r"pre-cleaned substrate|planar test substrate", "unspecified"),
)
_FREE = re.compile(r"free-?standing|freestanding|none|self-supported|collected on a sieve|lifted off|"
                   r"winding spool|mou?ld|printer platform|bulk hydrogel", re.I)


def substrate_ids(text):
    s = (text or "").strip().lower()
    if not s:
        return []
    found = []
    for pattern, sid in _SUB_RULES:
        if re.search(pattern, s) and sid not in found:
            if sid == "silicon" and "aptes" in s:
                continue
            if sid == "quartz" and "quartz_crystal" in found:
                continue
            if sid == "polymer_membrane" and "alumina_porous" in found:
                continue
            found.append(sid)
    if _FREE.search(s) and "free_standing" not in found:
        found.append("free_standing")
    return found or ["other_reported"]


def substrate_features(text, prefix="openD_"):
    ids = substrate_ids(text)
    mats = [SUBSTRATES[i] for i in ids if i in SUBSTRATES]
    out = {prefix + "free_standing": float("free_standing" in ids),
           prefix + "n_materials": float(len(mats)),
           prefix + "known": float(bool(mats) and any(i != "unspecified" for i in ids if i in SUBSTRATES))}
    for j, name in enumerate(SUBSTRATE_FIELDS):
        vals = [m[j] for m in mats if not (isinstance(m[j], float) and math.isnan(m[j]))]
        if not vals:
            continue
        if name == "log10_k_W_mK":
            vals = [math.log10(v) for v in vals]
        if name in ("conductive", "flexible", "porous", "polymer", "inorganic", "transparent"):
            out[prefix + name] = float(max(vals))
        else:
            out[prefix + name] = float(sum(vals) / len(vals))
    return out
