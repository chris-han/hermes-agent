"""Windows native assets retain supplied branding across build flavors and sizes."""
import io
from pathlib import Path

from PIL import Image

from scripts.windows_brand_icons import desktop_icon_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_windows_assets_use_supplied_art_at_every_native_size():
    original = Image.open(ROOT / 'apps/desktop/src/assets/semantier-logo.png').convert('RGBA')
    for kind in ['png', 'png_dark', 'girl_light', 'girl_dark']:
        for size in [16, 20, 44, 50, 150, 256, 512, 1024]:
            data = desktop_icon_bytes(ROOT, 'apps/desktop/assets/icon.png', kind, size)
            actual = Image.open(io.BytesIO(data)).convert('RGBA')
            assert actual.tobytes() == original.resize((size, size), Image.Resampling.LANCZOS).tobytes()
    data = desktop_icon_bytes(ROOT, 'apps/desktop/assets/icon.ico', 'ico', [16, 24, 32, 48, 64, 128, 256])
    icon = Image.open(io.BytesIO(data))
    assert icon.ico.sizes() == {(s, s) for s in [16, 24, 32, 48, 64, 128, 256]}
    assert icon.convert('RGBA').tobytes() == original.resize((256, 256), Image.Resampling.LANCZOS).tobytes()
    wide = Image.open(io.BytesIO(desktop_icon_bytes(ROOT, 'apps/desktop/assets/appx/Wide310x150Logo.png', 'wide', (310,150,100))))
    assert wide.size == (310,150)
    assert wide.crop((105,25,205,125)).tobytes() == original.resize((100,100),Image.Resampling.LANCZOS).tobytes()
    assert wide.getpixel((0,0))[3] == 0


def test_other_products_and_platform_resources_keep_their_owner():
    assert desktop_icon_bytes(ROOT, 'website/static/img/logo.png', 'png', 32) is None
    assert desktop_icon_bytes(ROOT, 'apps/bootstrap-installer/icons/icon.ico', 'ico', [32]) is None
    assert desktop_icon_bytes(ROOT, 'apps/desktop/assets/icon.icns', 'icns', None) is None
    assert desktop_icon_bytes(ROOT, 'apps/desktop/assets/icon-mac.png', 'png_mac', 1024) is None
