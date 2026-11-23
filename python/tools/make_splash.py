"""Generate the MPM Studio splash-screen image (gui/resources/splash.png, 600 x 400 px).

Only the language-neutral parts are drawn here: the background, the application icon and the name.
The tagline, the version and the start-up messages are drawn at run time (gui/splash.py) so they follow
the interface language. Any other PNG of at most 600 x 400 px can replace splash.png; larger images are
scaled down at run time.

    .venv\\Scripts\\python.exe python\\tools\\make_splash.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "resources")
W, H, K = 600, 400, 2                                   # final size, supersampling factor
FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
BANDS = ["#57106e", "#bc3754", "#e45a31", "#f98e09", "#fbc33a"]       # inferno, as in the icon


def _hex(c):
    return tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def _font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FONTS, name), size)
    except OSError:
        return ImageFont.load_default()


def main():
    w, h = W * K, H * K
    t = np.linspace(0, 1, h)[:, None, None]                              # vertical gradient
    top, bottom = np.array(_hex("#1d2635")), np.array(_hex("#0e131c"))
    img = Image.fromarray(np.broadcast_to(top + (bottom - top) * t, (h, w, 3)).astype("uint8"), "RGB").convert("RGBA")

    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))                       # faint map grid
    g = ImageDraw.Draw(grid)
    for x in range(0, w, 40 * K):
        g.line([(x, 0), (x, h)], fill=(255, 255, 255, 14), width=K)
    for y in range(0, h, 40 * K):
        g.line([(0, y), (w, y)], fill=(255, 255, 255, 14), width=K)
    img = Image.alpha_composite(img, grid)

    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))                       # warm glow behind the icon
    ImageDraw.Draw(glow).ellipse([40 * K, 60 * K, 270 * K, 300 * K], fill=_hex("#bc3754") + (90,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(45 * K)))

    icon = Image.open(os.path.join(RES, "mpm_studio.png")).convert("RGBA").resize((210 * K, 210 * K), Image.LANCZOS)
    img.alpha_composite(icon, (52 * K, 72 * K))

    d = ImageDraw.Draw(img)
    d.text((292 * K, 118 * K), "MPM Studio", font=_font("seguisb.ttf", 48 * K), fill=(255, 255, 255, 255))
    x0, x1, y0 = 296 * K, 560 * K, 196 * K                               # inferno accent bar under the name
    seg = (x1 - x0) / (len(BANDS) - 1)
    cols = np.array([_hex(c) for c in BANDS], float)
    for x in range(x0, x1):
        f = (x - x0) / seg
        i = min(int(f), len(BANDS) - 2)
        c = cols[i] + (cols[i + 1] - cols[i]) * (f - i)
        d.line([(x, y0), (x, y0 + 4 * K)], fill=tuple(int(v) for v in c) + (255,))
    d.line([(0, h - 44 * K), (w, h - 44 * K)], fill=(255, 255, 255, 22), width=K)   # status strip separator

    out = img.resize((W, H), Image.LANCZOS).convert("RGB")
    meta = PngInfo()
    meta.add_text("mpm-splash", "builtin")          # gui/splash.py draws the tagline / version only on this image
    out.save(os.path.join(RES, "splash.png"), optimize=True, pnginfo=meta)
    print("wrote", os.path.join(RES, "splash.png"), out.size)


if __name__ == "__main__":
    main()
