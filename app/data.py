# app/data.py — ФРАГМЕНТ (замена build_image_index + хелпер)
import os
from urllib.parse import urlparse

def _extract_code_from_url(url: str) -> str:
    """
    Достаём код из имени файла в URL:
    https://i.ibb.co/ID/UZ000346.jpg -> uz000346
    Регистронезависимо, только буквы+цифры.
    """
    try:
        path = urlparse((url or "").strip()).path  # /ID/UZ000346.jpg
        if not path:
            return ""
        base = os.path.basename(path)              # UZ000346.jpg
        name, _ext = os.path.splitext(base)        # UZ000346
        _raw, sq = _norm_code(name)                # -> uz000346
        return sq
    except Exception:
        return ""

def build_image_index(df: pd.DataFrame) -> dict[str, str]:
    """
    Два режима:
    - IMAGE_STRICT=1: строим индекс по имени файла (UZ000346.jpg -> uz000346).
      Работает для прямых ссылок i.ibb.co/.../CODE.ext. Google Drive почти всегда
      отфильтруется (в URL нет кода) — так и задумано.
    - IMAGE_STRICT=0: старое поведение — берём URL из текущей строки по её 'код'.
    """
    if "image" not in df.columns:
        logger.info("image-index: колонки 'image' нет — индекс пуст.")
        return {}

    index: dict[str, str] = {}
    added = 0

    try:
        if IMAGE_STRICT:
            # Строгий режим: парсим код ИЗ URL (имя файла). Не привязываемся к значению 'код' в строке.
            for _i, row in df.iterrows():
                url = str(row.get("image", "")).strip()
                if not url:
                    continue
                code_from_url = _extract_code_from_url(url)
                if not code_from_url:
                    continue
                if code_from_url not in index:
                    index[code_from_url] = url
                    added += 1
            logger.info(f"image-index[STRICT(FILENAME)]: добавлено {added} из {len(df)} строк")
            return index
        else:
            # Мягкий режим: маппинг 'код' -> 'image' из строки (как раньше).
            for _i, row in df.iterrows():
                code_val = str(row.get("код", "")).strip()
                url = str(row.get("image", "")).strip()
                if not code_val or not url:
                    continue
                raw, sq = _norm_code(code_val)
                # не перетираем уже записанное
                if raw and raw not in index:
                    index[raw] = url
                if sq and sq not in index:
                    index[sq] = url
                    # считаем добавленным, если хотя бы одна запись появилась
                    added += 1
            logger.info(f"image-index[ROW-MAP]: добавлено {added} из {len(df)} строк")
            return index
    except Exception as e:
        logger.warning(f"image-index: ошибка построения ({e}), индекс будет пуст.")
        return {}
