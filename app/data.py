# ВСТАВИТЬ/ОБНОВИТЬ рядом с другими утилитами
def _norm_code(c: str) -> tuple[str, str]:
    raw = (c or "").strip().lower()
    squash_ = re.sub(r'[\W_]+', '', raw, flags=re.UNICODE)
    return raw, squash_

def _url_has_code(url: str, code_raw: str, code_sq: str) -> bool:
    """
    Истина, если код встречается где-угодно в URL (включая query).
    Сравниваем и обычную строку, и «сквошнутую» без разделителей.
    """
    s = (url or "").strip().lower()
    if not s:
        return False
    s_sq = re.sub(r'[\W_]+', '', s, flags=re.UNICODE)
    return (code_raw and code_raw in s) or (code_sq and code_sq in s_sq)

def normalize_drive_url(url: str) -> str:
    m = re.search(r'drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))', url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

async def resolve_image_url_async(u: str) -> str:
    """
    Нормализуем drive-ссылки (превращаем в прямой download),
    ничего не скачиваем и не дергаем внешние сайты.
    """
    u = (u or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    return u

def build_image_index(df: pd.DataFrame) -> dict[str, str]:
    """
    Индекс «код -> url» строим ТОЛЬКО если код встречается в самом URL
    (в пути или в query). Если нет — пропускаем. Никакого фолбэка.
    """
    if "image" not in df.columns or "код" not in df.columns:
        return {}

    index: dict[str, str] = {}
    total = 0
    matched = 0
    for _, row in df.iterrows():
        total += 1
        code_val = str(row.get("код", "")).strip().lower()
        url = str(row.get("image", "")).strip()
        if not code_val or not url:
            continue

        raw, sq = _norm_code(code_val)
        if _url_has_code(url, raw, sq):
            # нормализуем только drive, остальное трогаем минимально
            url_norm = normalize_drive_url(url)
            if raw and raw not in index:
                index[raw] = url_norm
            if sq and sq not in index:
                index[sq] = url_norm
            matched += 1
        # иначе – пропускаем (не шлём фото)

    logger.info(f"image-index: совпадений по URL={matched} из {total}")
    return index

async def find_image_by_code_async(code: str) -> str:
    if not code or _image_index is None:
        return ""
    raw, sq = _norm_code(code)
    return _image_index.get(raw) or _image_index.get(sq, "") or ""
