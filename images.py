import re
import aiohttp
import io
import logging

logger = logging.getLogger("images")

def normalize_drive_url(url: str) -> str:
    m = re.search(r'drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))', url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

async def resolve_ibb_direct_async(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=12) as resp:
                if resp.status != 200:
                    return url
                html = await resp.text()
        m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        return m.group(1) if m else url
    except Exception as e:
        logger.warning(f"resolve_ibb_direct_async fail: {e}")
        return url

async def resolve_image_url_async(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    if re.match(r"^https?://(www\.)?ibb\.co/", u, re.I):
        return await resolve_ibb_direct_async(u)
    return u

async def download_image_async(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=12) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                if len(data) > 5_000_000:
                    return None
                bio = io.BytesIO(data)
                bio.name = "image"
                return bio
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return None