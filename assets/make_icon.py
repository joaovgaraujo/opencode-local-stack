# make_icon.py - generates the OpenCode Local icon set.
# Design: dark rounded tile, green terminal prompt ">_" (a coding agent in a
# terminal) and a small green dot in the corner (runs locally, on your PC).
# Outputs:
#   assets/logo.png             512px logo
#   assets/opencode-local.ico   multi-size Windows icon (16..256)
# Run: python assets/make_icon.py   (needs Pillow)
import os

from PIL import Image, ImageDraw

S = 1024  # master canvas, downscaled for exports (supersampling)

BG = (24, 27, 38, 255)        # dark slate tile
PROMPT = (74, 222, 128, 255)  # green prompt
CURSOR = (240, 244, 255, 255) # near-white cursor bar
DOT = (74, 222, 128, 255)

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=BG)

# ">" chevron: a thick polyline with round joints
w = int(S * 0.11)
pts = [(S * 0.22, S * 0.30), (S * 0.47, S * 0.50), (S * 0.22, S * 0.70)]
d.line(pts, fill=PROMPT, width=w, joint="curve")
for x, y in (pts[0], pts[2]):
    d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=PROMPT)

# "_" cursor
d.rounded_rectangle([S * 0.52, S * 0.64, S * 0.80, S * 0.64 + w], radius=w // 2, fill=CURSOR)

# "local" dot, top right
r = S * 0.065
cx, cy = S * 0.78, S * 0.24
d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DOT)

here = os.path.dirname(os.path.abspath(__file__))
img.resize((512, 512), Image.LANCZOS).save(os.path.join(here, "logo.png"))
sizes = [(n, n) for n in (16, 20, 24, 32, 40, 48, 64, 128, 256)]
img.resize((256, 256), Image.LANCZOS).save(os.path.join(here, "opencode-local.ico"), sizes=sizes)
print("wrote assets/logo.png and assets/opencode-local.ico")
