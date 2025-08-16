import gspread
import pandas as pd
from pandas import DataFrame
import re
from typing import List, Set, Tuple, Dict, Any
import logging
from config import SPREADSHEET_URL, SHEET_NAME, get_creds
from utils import now_local_str

logger = logging.getLogger("gsheets")

def get_gs_client():
    return gspread.authorize(get_creds())

def _open_data_worksheet(client):
    sh = client.open_by_url(SPREADSHEET_URL)
    if SHEET_NAME:
        try:
            return sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            logger.warning(f"Лист {SHEET_NAME!r} не найден, fallback на sheet1")
    return sh.sheet1

def load_data_blocking() -> list[dict]:
    client = get_gs_client()
    ws = _open_data_worksheet(client)
    return ws.get_all_records()

def load_users_from_sheet() -> Tuple[Set[int], Set[int], Set[int]]:
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("Пользователи")
    except gspread.WorksheetNotFound:
        try:
            ws = sh.worksheet("Users")
        except gspread.WorksheetNotFound:
            logger.info("Лист 'Пользователи' не найден — доступ разрешён всем.")
            return set(), set(), set()

    headers_raw = ws.row_values(1) or []
    if not headers_raw:
        logger.info("В листе 'Пользователи' пустая шапка — доступ разрешён всем.")
        return set(), set(), set()

    norm_headers: List[str] = []
    seen: Set[str] = set()
    for i, h in enumerate(headers_raw, start=1):
        name = (h or "").strip()
        if not name:
            name = f"col_{i}"
        lname = re.sub(r"\s+", " ", name.lower()).strip()
        base = lname
        k = 1
        while lname in seen:
            k += 1
            lname = f"{base}_{k}"
        seen.add(lname)
        norm_headers.append(lname)

    try:
        rows = ws.get_all_records(expected_headers=norm_headers)
    except Exception as e:
        logger.warning(f"get_all_records с expected_headers не сработал ({e}), переходим на ручной парсинг")
        values = ws.get_all_values()
        data_rows = values[1:] if len(values) > 1 else []
        rows = []
        for r in data_rows:
            padded = (r + [''] * (len(norm_headers) - len(r)))[:len(norm_headers)]
            rows.append({norm_headers[i]: padded[i] for i in range(len(norm_headers))})

    if not rows:
        logger.info("Лист 'Пользователи' пуст — доступ разрешён всем.")
        return set(), set(), set()

    def _to_int_or_none(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            s = str(x).strip()
            if not s:
                return None
            m = re.search(r"-?\d+", s)
            return int(m.group(0)) if m else None
        except Exception:
            return None

    def _truthy_local(x) -> bool:
        s = str(x).strip().lower()
        return (
            s in {"1", "true", "yes", "y", "да", "истина", "ok", "ок",
                  "allowed", "разрешен", "разрешено", "доступ",
                  "admin", "админ", "ban", "blocked", "запрет"}
            or (s.isdigit() and int(s) > 0)
        )

    allowed, admins, blocked = set(), set(), set()
    for row in rows:
        r = {str(k).strip().lower(): v for k, v in row.items()}

        uid = (
            _to_int_or_none(r.get("user_id"))
            or _to_int_or_none(r.get("userid"))
            or _to_int_or_none(r.get("id"))
            or _to_int_or_none(r.get("uid"))
            or _to_int_or_none(r.get("телеграм id"))
            or _to_int_or_none(r.get("пользователь"))
            or _to_int_or_none(r.get("user"))
        )
        if not uid:
            continue

        role = str(r.get("role") or r.get("роль") or "").strip().lower()
        is_admin = role in {"admin", "админ", "administrator", "администратор"} or _truthy_local(r.get("admin"))
        is_blocked = _truthy_local(r.get("blocked") or r.get("ban") or r.get("запрет"))
        is_allowed = _truthy_local(r.get("allowed") or r.get("доступ") or (not role or role == "user"))

        if is_blocked:
            blocked.add(uid)
        if is_admin:
            admins.add(uid); is_allowed = True
        if is_allowed and not is_blocked:
            allowed.add(uid)

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked

def save_issue_to_sheet_blocking(user, part: Dict[str, Any], quantity, comment: str):
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("История")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="История", rows=1000, cols=12)
        ws.append_row(["Дата", "ID", "Имя", "Тип", "Наименование", "Код", "Количество", "Коментарий"])

    headers_raw = ws.row_values(1)
    headers = [h.strip() for h in headers_raw]
    norm = [h.lower() for h in headers]

    full_name = f"{(user.first_name or '').strip()} {(user.last_name or '').strip()}".strip()
    display_name = full_name or (f"@{user.username}" if user.username else str(user.id))
    ts = now_local_str()

    values_by_key = {
        "дата": ts, "timestamp": ts,
        "id": user.id, "user_id": user.id,
        "имя": display_name, "name": display_name,
        "тип": str(part.get("тип", "")), "type": str(part.get("тип", "")),
        "наименование": str(part.get("наименование", "")), "name_item": str(part.get("наименование", "")),
        "код": str(part.get("код", "")), "code": str(part.get("код", "")),
        "数量": str(quantity), "количество": str(quantity), "qty": str(quantity),
        "коментарий": comment or "", "комментарий": comment or "", "comment": comment or "",
    }

    row = [values_by_key.get(hn, "") for hn in norm]
    ws.append_row(row, value_input_option="USER_ENTERED")
    logger.info("💾 Списание записано в 'История'")