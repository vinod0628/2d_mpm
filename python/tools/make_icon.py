"""Generate the MPM Studio application icon (PNG + multi-size Windows ICO).

Motif: a prospectivity "hotspot" (banded posterior-probability contours in the inferno palette used by
the app) on a map grid, with a faceted mineral crystal at its centre.

    .venv\\Scripts\\python.exe python\\tools\\make_icon.py
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "resources")
S = 1024                                                      # master size (downscaled with antialiasing)

# inferno-like class colours, low -> high prospectivity
BANDS = ["#2a0b4f", "#57106e", "#8a226a", "#bc3754", "#e45a31", "#f98e09", "#fbc33a", "#fcffa4"]


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def hotspot(size):
    """Banded, slightly irregular prospectivity field -> RGBA image."""
    y, x = np.mgrid[0:size, 0:size] / size
    f = np.zeros((size, size))
    for cx, cy, sx, sy, w in [(0.47, 0.52, 0.22, 0.17, 1.0), (0.66, 0.36, 0.11, 0.13, 0.55),
                              (0.30, 0.70, 0.12, 0.09, 0.45), (0.58, 0.68, 0.09, 0.08, 0.35)]:
        f += w * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2) / 2)
    f += 0.04 * np.sin(9 * x + 3 * y) * np.cos(7 * y - 2 * x)        # geological texture
    f = (f - f.min()) / (f.max() - f.min())
    levels = np.clip((f ** 0.85 * len(BANDS)).astype(int), 0, len(BANDS) - 1)
    rgb = np.array([_hex(c) for c in BANDS], dtype=np.uint8)[levels]
    alpha = np.clip((f - 0.10) / 0.18, 0, 1)                           # fade into the background
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    img = Image.fromarray(rgba, "RGBA")
    # thin contour lines between classes
    edges = np.zeros((size, size), bool)
    edges[:-1, :] |= levels[:-1, :] != levels[1:, :]
    edges[:, :-1] |= levels[:, :-1] != levels[:, 1:]
    edges &= f > 0.18
    line = Image.fromarray((edges * 110).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(3))
    dark = Image.new("RGBA", (size, size), (20, 10, 30, 0))
    dark.putalpha(line)
    img.alpha_composite(dark)
    return img


def crystal(draw, cx, cy, r):
    """Faceted gem (brilliant-cut silhouette) centred at (cx, cy)."""
    top = cy - 0.55 * r
    girdle = cy - 0.12 * r
    tip = cy + 0.95 * r
    tw, gw = 0.62 * r, 1.0 * r
    p = dict(tl=(cx - tw, top), tr=(cx + tw, top), gl=(cx - gw, girdle), gr=(cx + gw, girdle), tip=(cx, tip),
             tm=(cx, top), gm1=(cx - 0.42 * r, girdle), gm2=(cx + 0.42 * r, girdle))
    facets = [  # (polygon, fill)
        ([p["tl"], p["tm"], p["gm1"], p["gl"]], (196, 236, 255)),
        ([p["tm"], p["tr"], p["gr"], p["gm2"]], (130, 200, 245)),
        ([p["tm"], p["gm2"], p["gm1"]], (235, 250, 255)),
        ([p["gl"], p["gm1"], p["tip"]], (120, 185, 235)),
        ([p["gm1"], p["gm2"], p["tip"]], (165, 220, 250)),
        ([p["gm2"], p["gr"], p["tip"]], (78, 140, 210)),
    ]
    outline = (16, 32, 60)
    lw = max(2, int(r * 0.045))
    # soft shadow
    for poly, fill in facets:
        draw.polygon([(x + r * 0.06, y + r * 0.08) for x, y in poly], fill=(0, 0, 0, 90))
    for poly, fill in facets:
        draw.polygon(poly, fill=fill + (255,))
    for poly, _ in facets:
        draw.line(poly + [poly[0]], fill=outline, width=lw, joint="curve")
    # sparkle
    sx, sy, sr = cx - 0.38 * r, top + 0.14 * r, 0.16 * r
    draw.polygon([(sx, sy - sr), (sx + sr * 0.25, sy - sr * 0.25), (sx + sr, sy), (sx + sr * 0.25, sy + sr * 0.25),
                  (sx, sy + sr), (sx - sr * 0.25, sy + sr * 0.25), (sx - sr, sy), (sx - sr * 0.25, sy - sr * 0.25)],
                 fill=(255, 255, 255, 235))


def make(size=S) -> Image.Image:
    radius = int(size * 0.22)
    # background: deep blue-grey vertical gradient
    g = np.linspace(0, 1, size)[:, None]
    top, bot = np.array(_hex("#26303f")), np.array(_hex("#11161f"))
    bg = (top * (1 - g) + bot * g).astype(np.uint8)
    bg = np.repeat(bg[:, None, :], size, axis=1).reshape(size, size, 3)
    img = Image.fromarray(bg, "RGB").convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    # map grid
    step = size // 8
    for k in range(1, 8):
        d.line([(k * step, 0), (k * step, size)], fill=(255, 255, 255, 18), width=max(1, size // 256))
        d.line([(0, k * step), (size, k * step)], fill=(255, 255, 255, 18), width=max(1, size // 256))
    img.alpha_composite(hotspot(size))
    # target ring around the hotspot (exploration target)
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy, rr = size * 0.47, size * 0.52, size * 0.30
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 255, 255, 150), width=max(2, size // 90))
    for a in (0, 90, 180, 270):
        ax, ay = math.cos(math.radians(a)), math.sin(math.radians(a))
        d.line([(cx + ax * rr * 0.86, cy + ay * rr * 0.86), (cx + ax * rr * 1.14, cy + ay * rr * 1.14)],
               fill=(255, 255, 255, 170), width=max(2, size // 70))
    crystal(d, cx, cy, size * 0.17)
    # rounded tile + subtle border
    mask = rounded_mask(size, radius)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    ImageDraw.Draw(out).rounded_rectangle([1, 1, size - 2, size - 2], radius=radius,
                                          outline=(255, 255, 255, 40), width=max(1, size // 128))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    big = make(S)
    png = os.path.join(OUT_DIR, "mpm_studio.png")
    big.resize((512, 512), Image.LANCZOS).save(png)
    ico = os.path.join(OUT_DIR, "mpm_studio.ico")
    sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    big.resize((256, 256), Image.LANCZOS).save(ico, sizes=[(s, s) for s in sizes])
    preview = Image.new("RGBA", (sum(sizes[-6:]) + 20 * 6, 260), (43, 45, 48, 255))
    x = 10
    for s in [24, 32, 48, 64, 128, 256]:
        preview.alpha_composite(big.resize((s, s), Image.LANCZOS), (x, 130 - s // 2))
        x += s + 20
    preview.save(os.path.join(OUT_DIR, "mpm_studio_preview.png"))
    print(png, ico, sep="\n")


if __name__ == "__main__":
    main()
