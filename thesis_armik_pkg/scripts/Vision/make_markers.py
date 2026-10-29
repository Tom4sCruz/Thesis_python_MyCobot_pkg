#!/usr/bin/env python3
"""
VISION: printable marker sheet -> markers_sheet.pdf (one A4 page)
=================================================================

Everything on ONE A4 page, drawn at exactly 300 dpi:
  * six cube-face tiles (IDs 0-5): a 3.0 cm marker centred on a 3.4 cm white
    tile -- cut along the dashed lines and glue one tile, centred, on each face
    of the 3.5 cm cube. Which face gets which ID, and how each tile is rotated,
    does not matter (see _vision.py).
  * BASE (ID 10) and TARGET (ID 11): 8 cm markers, each with a 1 cm white
    border inside its dashed cut line.
  * a 10 cm scale bar.

Printing: "Actual size" / 100 % -- never "fit to page". Then MEASURE the
scale bar (10.0 cm) and one black square of each size; if they are off, pass
the measured values to the other scripts (--cube-marker-cm / --big-marker-cm)
or fix the printer setting.

Laying out the BASE marker: flat on the table next to the robot, with the
"robot +X" arrow pointing along the robot's +x axis (straight ahead of J1 at
0 deg). Measure the marker centre's offset from the J1 axis into
_vision.BASE_MARKER_POSE; calibrate_base.py refines it.

The quiet zone (white border) around each black square is what lets the
detector find it -- don't trim into it, and keep the tiles on a white or
light background.

    python3 scripts/Vision/make_markers.py              # -> markers_sheet.pdf
    python3 scripts/Vision/make_markers.py --png        # also markers_sheet.png

Needs Pillow (for the PDF) and OpenCV.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

import _vision as V


# =============================================================================
# CONSTANTS  (cm unless noted)
# =============================================================================

DPI = 300
PAGE_W_CM, PAGE_H_CM = 21.0, 29.7          # A4 portrait
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markers_sheet.pdf")

BIG_BORDER_CM = 1.0                        # white margin around an 8 cm marker
LEFT_X_CM = 1.2                            # left edge of the BASE / TARGET column
TOP_Y_CM = 2.6                             # top of the marker area (title above)
BIG_GAP_CM = 1.6                           # vertical gap between BASE and TARGET (labels)
TILE_COL_X_CM = 12.2                       # left edge of the cube-tile block
TILE_GAP_X_CM = 1.0
TILE_GAP_Y_CM = 1.4                        # room for the ID label under each tile

CUT_GREY = 150                             # dashed cut lines
LABEL_GREY = 90
HINT_GREY = 185                            # text inside a white border: light enough that
                                           # the detector still reads the border as white
DASH_PX = 18


# =============================================================================
# DRAWING
# =============================================================================

def px(cm):
    return int(round(cm * DPI / 2.54))


def marker_image(dictionary, marker_id, side_px):
    """Marker at exactly side_px, with sharp cell edges (nearest-neighbour upscale)."""
    cells = dictionary.markerSize + 2              # data bits + the black border
    small = cv2.aruco.generateImageMarker(dictionary, marker_id, cells)
    return cv2.resize(small, (side_px, side_px), interpolation=cv2.INTER_NEAREST)


def dashed_rect(page, x0, y0, x1, y1, grey=CUT_GREY):
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = max(abs(bx - ax), abs(by - ay))
        for s in range(0, length, 2 * DASH_PX):
            e = min(s + DASH_PX, length)
            p = (ax + (bx - ax) * s // length, ay + (by - ay) * s // length)
            q = (ax + (bx - ax) * e // length, ay + (by - ay) * e // length)
            cv2.line(page, p, q, grey, 2)


def text(page, s, x, y, scale=1.6, grey=LABEL_GREY, thick=3):
    cv2.putText(page, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, grey, thick, cv2.LINE_AA)


def paste_marker(page, img, x, y):
    page[y:y + img.shape[0], x:x + img.shape[1]] = img


def big_marker(page, dictionary, marker_id, x_cm, y_cm, label, big_cm):
    """One 8 cm marker with its white border and cut line; returns the cut box bottom (cm)."""
    border = px(BIG_BORDER_CM)
    side = px(big_cm)
    x0, y0 = px(x_cm), px(y_cm)
    paste_marker(page, marker_image(dictionary, marker_id, side), x0 + border, y0 + border)
    x1, y1 = x0 + side + 2 * border, y0 + side + 2 * border
    dashed_rect(page, x0, y0, x1, y1)
    text(page, label, x0, y1 + px(0.7), scale=1.5)
    return y_cm + big_cm + 2 * BIG_BORDER_CM


def base_arrow(page, x_cm, y_cm, big_cm):
    """'robot +X' arrow in the BASE marker's top white border, pointing up the page."""
    cx = px(x_cm + BIG_BORDER_CM + big_cm / 2.0)
    top, bottom = px(y_cm + 0.15), px(y_cm + BIG_BORDER_CM - 0.15)
    cv2.arrowedLine(page, (cx - px(1.6), bottom), (cx - px(1.6), top), HINT_GREY, 4,
                    cv2.LINE_AA, tipLength=0.35)
    text(page, "robot +X", cx - px(1.2), bottom - px(0.2), scale=1.3, grey=HINT_GREY, thick=3)


def cube_tiles(page, dictionary, cube_cm, tile_cm):
    """2 columns x 3 rows of tiles, IDs 0-5."""
    tile, side = px(tile_cm), px(cube_cm)
    pad = (tile - side) // 2
    for k, marker_id in enumerate(V.CUBE_FACE_IDS):
        col, row = k % 2, k // 2
        x0 = px(TILE_COL_X_CM + col * (tile_cm + TILE_GAP_X_CM))
        y0 = px(TOP_Y_CM + row * (tile_cm + TILE_GAP_Y_CM))
        paste_marker(page, marker_image(dictionary, marker_id, side), x0 + pad, y0 + pad)
        dashed_rect(page, x0, y0, x0 + tile, y0 + tile)
        text(page, f"cube id {marker_id}", x0, y0 + tile + px(0.6), scale=1.1, thick=2)
    y_end = TOP_Y_CM + 3 * (tile_cm + TILE_GAP_Y_CM)
    text(page, f"tiles: {tile_cm:g} cm,", px(TILE_COL_X_CM), px(y_end + 0.4), scale=1.1, thick=2)
    text(page, f"marker {cube_cm:g} cm", px(TILE_COL_X_CM), px(y_end + 0.95), scale=1.1, thick=2)


def scale_bar(page, x_cm, y_cm, length_cm=10):
    x0, y = px(x_cm), px(y_cm)
    cv2.line(page, (x0, y), (px(x_cm + length_cm), y), 0, 4)
    for i in range(length_cm + 1):
        h = px(0.5 if i % 5 == 0 else 0.3)
        cv2.line(page, (px(x_cm + i), y), (px(x_cm + i), y - h), 0, 3)
    text(page, f"{length_cm} cm -- measure this after printing", x0, y + px(0.8), scale=1.2, grey=0)


def render_page(cube_cm, tile_cm, big_cm):
    page = np.full((px(PAGE_H_CM), px(PAGE_W_CM)), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, V.ARUCO_DICT))
    text(page, f"armik markers  {V.ARUCO_DICT}   print at 100 % (Actual size), A4",
         px(LEFT_X_CM), px(1.3), scale=1.5, grey=0)

    y = big_marker(page, dictionary, V.BASE_ID, LEFT_X_CM, TOP_Y_CM,
                   f"BASE  id {V.BASE_ID}  ({big_cm:g} cm)", big_cm)
    base_arrow(page, LEFT_X_CM, TOP_Y_CM, big_cm)
    y = big_marker(page, dictionary, V.TARGET_ID, LEFT_X_CM, y + BIG_GAP_CM,
                   f"TARGET  id {V.TARGET_ID}  ({big_cm:g} cm)", big_cm)
    cube_tiles(page, dictionary, cube_cm, tile_cm)
    scale_bar(page, LEFT_X_CM, y + 1.9)
    return page


def self_check(page):
    """Detect every marker in the rendered page (downscaled ~ a camera view)."""
    small = cv2.resize(page, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    found = set(V.detect(V.make_detector(), cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)))
    want = set(V.CUBE_FACE_IDS) | {V.BASE_ID, V.TARGET_ID}
    missing = sorted(want - found)
    if missing:
        print(f"SELF-CHECK FAIL: markers not detected in the rendered page: {missing}")
        return False
    print(f"SELF-CHECK PASS: all {len(want)} markers detected ({sorted(found)})")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Write the A4 ArUco marker sheet (PDF, 300 dpi).")
    ap.add_argument("--out", default=OUT_PATH, help="output PDF")
    ap.add_argument("--png", action="store_true", help="also write a PNG next to the PDF")
    ap.add_argument("--cube-marker-cm", type=float, default=V.CUBE_MARKER_CM)
    ap.add_argument("--tile-cm", type=float, default=V.CUBE_TILE_CM)
    ap.add_argument("--big-marker-cm", type=float, default=V.BIG_MARKER_CM)
    args = ap.parse_args()

    if args.cube_marker_cm >= args.tile_cm:
        ap.error("--cube-marker-cm must be smaller than --tile-cm (the white border is needed)")
    tile_block_right = TILE_COL_X_CM + 2 * args.tile_cm + TILE_GAP_X_CM
    big_right = LEFT_X_CM + args.big_marker_cm + 2 * BIG_BORDER_CM
    if big_right > TILE_COL_X_CM - 0.5 or tile_block_right > PAGE_W_CM - 0.5:
        ap.error("these sizes do not fit the A4 layout")

    page = render_page(args.cube_marker_cm, args.tile_cm, args.big_marker_cm)
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is needed for the PDF: pip install Pillow")
    Image.fromarray(page).save(args.out, "PDF", resolution=DPI)
    print(f"Wrote {args.out}  (A4, {DPI} dpi). Print at 100 % / Actual size.")
    if args.png:
        png = os.path.splitext(args.out)[0] + ".png"
        Image.fromarray(page).save(png, dpi=(DPI, DPI))
        print(f"Wrote {png}")
    return 0 if self_check(page) else 1


if __name__ == "__main__":
    sys.exit(main())
