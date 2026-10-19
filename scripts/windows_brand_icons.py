"""Windows Desktop artwork from the same supplied PNG as organization fallback branding."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def desktop_icon_bytes(source: Path, relative: str, kind: str, size: object) -> bytes | None:
    """Override Windows/desktop PNG assets; leave other product/platform artwork owned upstream."""
    if not relative.startswith('apps/desktop/') or kind not in {
        'png', 'png_dark', 'ico', 'ico_dark', 'wide', 'girl_light', 'girl_dark',
    }:
        return None
    logo = source / 'apps/desktop/src/assets/semantier-logo.png'
    # Upstream artwork-only fixtures/products omit the Desktop branding asset.
    if not logo.is_file():
        return None
    with Image.open(logo) as opened:
        image = opened.convert('RGBA')
    output = io.BytesIO()
    if kind in {'ico', 'ico_dark'}:
        largest = max(size)
        image.resize((largest, largest), Image.Resampling.LANCZOS).save(
            output, 'ICO', sizes=[(s, s) for s in size],
        )
    elif kind == 'wide':
        width, height, tile = size
        canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS),
                     ((width - tile) // 2, (height - tile) // 2))
        canvas.save(output, 'PNG')
    else:
        image.resize((size, size), Image.Resampling.LANCZOS).save(output, 'PNG')
    return output.getvalue()
