"""Render a test-only PNG of the *actual* Semantier Rich CLI banner.

This is a Linux-hosted renderer snapshot with Windows inputs, NOT an image of a
packaged Windows app. No application state or installed Windows files are changed.
"""
from __future__ import annotations
import io
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes_cli import banner

os.environ["SEMANTIER_CLI_BRAND"] = "1"
os.environ["SEMANTIER_CLI_VERSION"] = "0.0.77"
os.environ["HERMES_FAST_STARTUP_BANNER"] = "0"

out = io.StringIO()
console = Console(file=out, width=128, color_system="truecolor", force_terminal=True, no_color=False)
tool_names = ["browser_exec", "clarify", "execute_code", "manage_connections",
              "delegate_task", "patch", "read_file", "search_files", "write_file",
              "memory", "tool_call", "tool_describe", "tool_search"]
tool_map = {
    "browser_exec":"browser","clarify":"clarify","execute_code":"code_execution",
    "manage_connections":"connections","delegate_task":"delegation",
    "patch":"file","read_file":"file","search_files":"file","write_file":"file",
    "memory":"memory","tool_call":"other","tool_describe":"other","tool_search":"other"
}
with patch.object(banner.shutil, "get_terminal_size", return_value=SimpleNamespace(columns=128)), \
     patch.object(banner, "_mcp_configured", return_value=False), \
     patch.object(banner, "get_update_result", return_value=0):
    banner.build_welcome_banner(
        console,
        model="dashscope-qwen3.5-plus",
        cwd="C:\\Users\\ChrisH",
        tools=[{"type":"function","function":{"name":n}} for n in tool_names],
        enabled_toolsets=list(set(tool_map.values())),
        session_id="1f1f799e",
        context_length=128000,
        provider="dashscope",
        get_toolset_for_tool=lambda name:tool_map.get(name,"other"),
        availability={"unavailable_toolsets":[],"lazy_tools":[],"disabled_tools":[]},
        skills_by_category={"legal":["contract_compliance","tender_review"],
                            "research":["evidence_trace","context_graph"]},
    )
raw = out.getvalue()
assert "Semantier Agent v0.0.77" in raw
assert "Nous Research" not in raw
assert "Hermes Agent" not in raw

# Parse the ANSI output emitted by the actual CLI Rich renderer. Only the
# output-to-PNG step is custom; branding, layout and text come from production.
esc = re.compile(r"\x1b\[([0-9;]*)m")
ansi_color = {
    30:(10,20,10),31:(240,100,90),32:(95,185,110),33:(255,196,0),
    37:(231,235,217),90:(115,125,108),93:(255,221,0)
}
rows=[]
fg=(221,231,206)
for line in raw.splitlines():
    pieces=[]
    pos=0
    fg=(221,231,206)
    for m in esc.finditer(line):
        if m.start()>pos:
            pieces.append((line[pos:m.start()],fg))
        nums=[int(q) for q in m.group(1).split(";") if q]
        i=0
        while i<len(nums):
            n=nums[i]
            if n==0: fg=(221,231,206)
            elif n in ansi_color:fg=ansi_color[n]
            elif n==38 and i+4<len(nums) and nums[i+1]==2:
                fg=tuple(nums[i+2:i+5]);i+=4
            elif n==39: fg=(221,231,206)
            i+=1
        pos=m.end()
    if pos<len(line):pieces.append((line[pos:],fg))
    rows.append(pieces)
fontpaths=[
"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
"/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
]
fontpath=next((x for x in fontpaths if Path(x).exists()),fontpaths[0])
font=ImageFont.truetype(fontpath,19)
cell=ImageFont.truetype(fontpath,19).getlength("M")
cn_candidates=list(Path("/usr/share/fonts").rglob("*CJK*Regular*.ttc"))
cn_font=ImageFont.truetype(str(cn_candidates[0]),19) if cn_candidates else font
from wcwidth import wcwidth
columns=128
padx=28
pady=26
rowh=26
w=int(columns*cell+padx*2)
h=int(len(rows)*rowh+pady*2+24)
canvas=Image.new("RGB",(w,h),(11,37,13))
d=ImageDraw.Draw(canvas)
for iy,pieces in enumerate(rows):
    x=padx;y=pady+iy*rowh
    for piece,color in pieces:
        for ch in piece:
            span=max(0,wcwidth(ch))
            if ch not in (" ","\r") and span:
                d.text((x,y),ch,fill=color,font=cn_font if ord(ch)>127 and "一"<=ch<="鿿" else font)
            x+=cell*span
# Test label outside CLI output, ensures the viewer knows the verification level.
label="RENDERER TEST  |  Linux host / Windows path fixture  |  Semantier v0.0.77"
d.text((padx,pady+len(rows)*rowh+6),label,fill=(145,163,133),font=ImageFont.truetype(fontpath,14))
target=ROOT/".ai-bridge"/"semantier-cli-test-preview.png"
target.parent.mkdir(exist_ok=True)
canvas.save(target)
print(f"PNG: {target.name} {w}x{h}; rendered_lines={len(rows)}")
print("--- CLI TEST OUTPUT ---")
print(esc.sub("", raw))
print("--- END CLI TEST OUTPUT ---")
