import re
from typing import Dict, Set, DefaultDict, List
from collections import defaultdict
from pandas import DataFrame

SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

def _norm_code(c: str) -> tuple[str, str]:
    raw = (c or "").strip().lower()
    squash = re.sub(r'[\W_]+', '', raw, flags=re.UNICODE)
    return raw, squash

def _filename_from_url(u: str) -> str:
    from urllib.parse import urlparse, parse_qs, unquote
    u = (u or "").strip()
    if not u:
        return ""
    p = urlparse(u)
    name = unquote(p.path.rsplit("/", 1)[-1] or "")
    if not name or "." not in name:
        qnames = []
        for v in parse_qs(p.query).values():
            qnames.extend(v)
        candidates = qnames + [p.fragment]
        for cand in candidates:
            if isinstance(cand, str):
                cand = unquote(cand)
                cand_name = cand.rsplit("/", 1)[-1]
                if "." in cand_name:
                    name = cand_name
                    break
    return name

def _tokens_from_filename(u: str) -> List[str]:
    name = _filename_from_url(u)
    if not name:
        return []
    base = name.rsplit(".", 1)[0]
    fused = re.sub(r"[\W_]+", "", base.lower(), flags=re.UNICODE)
    parts = re.split(r"[\W_]+", base.lower())
    parts = [p for p in parts if p]
    seen, out = set(), []
    for t in [fused] + parts:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def _fuse(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)

def build_search_index(df: DataFrame) -> Dict[str, Set[int]]:
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_FIELDS:
        if col not in df.columns:
            continue
        for idx, val in df[col].astype(str).str.lower().items():
            for t in re.findall(r'\w+', val):
                if t:
                    index[t].add(idx)
    return dict(index)

def build_image_index(df: DataFrame) -> Dict[str, str]:
    if "image" not in df.columns:
        return {}
    index: Dict[str, str] = {}
    for _, row in df.iterrows():
        url = str(row.get("image", "")).strip()
        if not url:
            continue
        tokens = _tokens_from_filename(url)
        for t in tokens:
            raw, sq = _norm_code(t)
            if raw:
                index.setdefault(raw, url)
            if sq and sq != raw:
                index.setdefault(sq, url)
        name_fused = tokens[:1]
        fused = name_fused[0] if name_fused else ""
        for field in ("код", "oem"):
            v = str(row.get(field, "")).strip().lower()
            if not v or not fused:
                continue
            raw, sq = _norm_code(v)
            if raw and raw in fused:
                index.setdefault(raw, url)
            if sq and sq in fused:
                index.setdefault(sq, url)
    return index