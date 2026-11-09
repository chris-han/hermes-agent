"""MSIX App Installer must show the Semantier icon without an opaque gray tile."""
from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "apps" / "desktop"


def test_app_installer_tile_background_is_transparent():
    result = subprocess.run(
        ["node", "-e", "console.log(JSON.stringify(require('./apps/desktop/electron-builder.config.cjs').msix))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    config = json.loads(result.stdout)
    assert config["backgroundColor"] == "transparent"


def test_production_msix_manifest_has_transparent_visual_elements():
    result = subprocess.run(
        ["node", "apps/desktop/scripts/gen-msix-manifest.mjs", "bundled", "x64"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    manifest = ET.fromstring(result.stdout)
    visual = manifest.find(".//{*}VisualElements")
    assert visual is not None
    assert visual.attrib["BackgroundColor"] == "transparent"
    assert visual.attrib["Square150x150Logo"].endswith("Square150x150Logo.png")


def test_windows_logo_assets_have_transparent_corners_and_full_size():
    for name in (
        "Square150x150Logo.png",
        "Square150x150Logo.scale-400.png",
        "Square44x44Logo.png",
        "Square44x44Logo.targetsize-256_altform-unplated.png",
        "Square44x44Logo.targetsize-256_altform-lightunplated.png",
    ):
        with Image.open(DESKTOP / "assets" / "appx" / name) as logo:
            image = logo.convert("RGBA")
        width, height = image.size
        alpha = image.getchannel("A")
        assert alpha.getpixel((0, 0)) == 0, name
        assert alpha.getpixel((width - 1, height - 1)) == 0, name
        bbox = alpha.getbbox()
        assert bbox is not None, name
        assert bbox[2] - bbox[0] >= width * 0.90, name
        assert bbox[3] - bbox[1] >= height * 0.90, name
