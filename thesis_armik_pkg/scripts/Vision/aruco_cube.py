#!/usr/bin/env python3
"""
VISION, STAGE 1: ArUco marker -> 3D cube overlay
================================================

Opens a camera (the laptop webcam by default), detects ArUco markers, estimates
each marker's pose and draws a semi-transparent 3D cube over the live image --
a visual check that the cube pose is computed correctly before any of it is
used to drive the arm.

Geometry: the marker is stuck on the cube's TOP face. The marker frame has its
origin at the marker centre, x/y in the marker plane and +z pointing out of the
face (toward the camera), so the cube is drawn from z = 0 down to z = -side.

The camera is assumed UNCALIBRATED: a pinhole model with focal length = image
width, principal point at the image centre and no distortion. That's good
enough to judge the overlay by eye; for accurate poses pass --calib with the
.npz written by calibrate_camera.py (`camera_matrix`, `dist_coeffs`, `image_size`).

Default marker: the printed University/Thesis/ArUco_marker_test_6x6_5cm.pdf
(6x6 dictionary, ID 0, 5 cm). It is larger than the 3.5 cm cubes, so the drawn
cube's top face sits centred INSIDE the marker -- expected, not a bug.

Units: cm throughout (marker length, cube side, translation).

Usage:
    python3 scripts/Vision/aruco_cube.py --print-marker 0     # write marker_0.png to print
    python3 scripts/Vision/aruco_cube.py                      # live overlay
    python3 scripts/Vision/aruco_cube.py --color red --alpha 0.7 --side 3.5
Keys: q / Esc quit, s save a snapshot PNG.

Needs OpenCV >= 4.7 (`pip install "opencv-python>=4.7"`, or `pip install -e .[vision]`).
Deliberately kept out of the `armik` package so armik stays OpenCV-free.
"""

import argparse
import sys
import time

import cv2
import numpy as np


# =============================================================================
# CONSTANTS  (each one can be overridden from the command line)
# =============================================================================

CUBE_SIDE_CM = 3.5               # edge length of the real cubes
CUBE_COLOR_BGR = (0, 200, 255)   # face colour (OpenCV is BGR!) -- or a name, see COLOR_NAMES
CUBE_ALPHA = 0.4                 # face opacity: 0 = invisible, 1 = opaque (edges always solid)
EDGE_DARKEN = 0.6                # edges are drawn in the face colour times this factor
EDGE_THICKNESS_PX = 2

MARKER_LENGTH_CM = 5.0           # side of the printed marker's outer BLACK square -- sets the
                                 # pose scale, so it must match the print (measure it! a
                                 # "fit to page" print setting changes it)
ARUCO_DICT = "DICT_6X6_50"       # University/Thesis/ArUco_marker_test_6x6_5cm.pdf: chev.me
                                 # 6x6, ID 0, 50 mm (6x6_50/100/250/1000 share the first IDs)
CAMERA_INDEX = 0                 # /dev/videoN
SHOW_AXES = True                 # draw the marker frame axes (x red, y green, z blue)
AXES_LENGTH_CM = 2.0

PRINT_MARKER_PX = 600            # size of the PNG written by --print-marker

COLOR_NAMES = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 80, 0),
    "yellow": (0, 230, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "orange": (0, 140, 255),
    "white": (255, 255, 255),
}

# Cube faces as indices into cube_vertices(): 0-3 top (marker plane), 4-7 bottom.
CUBE_FACES = (
    (0, 1, 2, 3),   # top
    (4, 5, 6, 7),   # bottom
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)
CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


# =============================================================================
# HELPERS
# =============================================================================

def parse_color(text):
    """'red' or 'B,G,R' -> (b, g, r) tuple of ints."""
    key = text.strip().lower()
    if key in COLOR_NAMES:
        return COLOR_NAMES[key]
    parts = key.split(",")
    if len(parts) == 3:
        try:
            bgr = tuple(int(p) for p in parts)
        except ValueError:
            bgr = None
        if bgr is not None and all(0 <= c <= 255 for c in bgr):
            return bgr
    raise argparse.ArgumentTypeError(
        f"colour must be one of {', '.join(COLOR_NAMES)} or 'B,G,R' (0-255), got {text!r}")


def get_dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise SystemExit(f"Unknown ArUco dictionary {name!r} (e.g. DICT_4X4_50, DICT_5X5_100)")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def build_camera_model(width, height, calib_path=None):
    """Camera matrix and distortion: from an .npz if given, else a rough pinhole guess."""
    if calib_path:
        data = np.load(calib_path)
        K = data["camera_matrix"].astype(np.float64)
        dist = data["dist_coeffs"].astype(np.float64)
        if "image_size" in data:
            cw, ch = (int(v) for v in data["image_size"])
            if (cw, ch) != (width, height):
                # Intrinsics are per-resolution; rescale (exact only if the aspect ratio matches).
                print(f"WARNING: calibrated at {cw}x{ch}, camera gives {width}x{height}; "
                      "rescaling the intrinsics -- recalibrate at this resolution for best results")
                K = K.copy()
                K[0, :] *= width / cw
                K[1, :] *= height / ch
        return K, dist
    f = float(width)
    K = np.array([[f, 0.0, width / 2.0],
                  [0.0, f, height / 2.0],
                  [0.0, 0.0, 1.0]])
    return K, np.zeros(5)


def marker_pose(corners, marker_length, K, dist):
    """Pose of one marker: (rvec, tvec) of the marker frame in the camera frame, or None.

    `corners` is the (1, 4, 2) array from the detector, ordered top-left,
    top-right, bottom-right, bottom-left -- the order SOLVEPNP_IPPE_SQUARE wants.
    """
    h = marker_length / 2.0
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 2).astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None


def cube_vertices(side):
    """8 cube corners in the marker frame: top face on the marker, body toward -z."""
    h = side / 2.0
    top = [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]]
    bottom = [[x, y, -side] for x, y, _ in top]
    return np.array(top + bottom)


def draw_cube(img, rvec, tvec, K, dist, side, color, alpha):
    """Draw a semi-transparent cube onto `img` in place."""
    verts = cube_vertices(side)
    pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2)

    # Painter's algorithm: fill the far faces first so near faces sit on top.
    R, _ = cv2.Rodrigues(rvec)
    depth = (verts @ R.T + tvec.reshape(1, 3))[:, 2]
    faces = sorted(CUBE_FACES, key=lambda f: -depth[list(f)].mean())

    if alpha > 0:
        overlay = img.copy()
        for face in faces:
            poly = np.round(pts2d[list(face)]).astype(np.int32)
            cv2.fillPoly(overlay, [poly], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)

    edge_color = tuple(int(c * EDGE_DARKEN) for c in color)
    for a, b in CUBE_EDGES:
        pa = tuple(int(v) for v in np.round(pts2d[a]))
        pb = tuple(int(v) for v in np.round(pts2d[b]))
        cv2.line(img, pa, pb, edge_color, EDGE_THICKNESS_PX, cv2.LINE_AA)


def put_text(img, text, org, scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def print_marker(dictionary, marker_id):
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, PRINT_MARKER_PX)
    border = PRINT_MARKER_PX // 6                      # white quiet zone, needed for detection
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    path = f"marker_{marker_id}.png"
    cv2.imwrite(path, img)
    print(f"Wrote {path}. After printing, measure the BLACK square and pass it as --marker-cm.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="ArUco marker -> 3D cube overlay on a live camera.")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX, help="camera index (/dev/videoN)")
    ap.add_argument("--side", type=float, default=CUBE_SIDE_CM, help="cube side length, cm")
    ap.add_argument("--color", type=parse_color, default=CUBE_COLOR_BGR,
                    help=f"cube colour: {', '.join(COLOR_NAMES)} or 'B,G,R'")
    ap.add_argument("--alpha", type=float, default=CUBE_ALPHA, help="face opacity, 0..1")
    ap.add_argument("--marker-cm", type=float, default=MARKER_LENGTH_CM,
                    help="printed marker black-square side, cm")
    ap.add_argument("--dict", default=ARUCO_DICT, help="ArUco dictionary name")
    ap.add_argument("--calib", default=None, help=".npz with camera_matrix and dist_coeffs")
    ap.add_argument("--no-axes", action="store_true", help="don't draw the marker axes")
    ap.add_argument("--print-marker", type=int, metavar="ID", default=None,
                    help="write a printable marker PNG and exit")
    args = ap.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        ap.error("--alpha must be between 0 and 1")

    dictionary = get_dictionary(args.dict)
    if args.print_marker is not None:
        print_marker(dictionary, args.print_marker)
        return 0

    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    window = "aruco_cube  (q/Esc quit, s snapshot)"
    K = dist = None
    fps, t_prev = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed", file=sys.stderr)
                return 1
            if K is None:
                h, w = frame.shape[:2]
                K, dist = build_camera_model(w, h, args.calib)
                print(f"Camera {args.camera}: {w}x{h}, "
                      f"{'calibrated (' + args.calib + ')' if args.calib else 'approximate intrinsics'}")

            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for c, marker_id in zip(corners, ids.flatten()):
                    pose = marker_pose(c, args.marker_cm, K, dist)
                    if pose is None:
                        continue
                    rvec, tvec = pose
                    draw_cube(frame, rvec, tvec, K, dist, args.side, args.color, args.alpha)
                    if not args.no_axes and SHOW_AXES:
                        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXES_LENGTH_CM)
                    x, y = c.reshape(4, 2).max(axis=0).astype(int)
                    put_text(frame, f"id {marker_id}  {np.linalg.norm(tvec):.1f} cm", (x + 5, y + 5))

            t_now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            n = 0 if ids is None else len(ids)
            put_text(frame, f"{fps:4.1f} fps   markers: {n}   side {args.side} cm   alpha {args.alpha}",
                     (10, 25))

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = time.strftime("aruco_cube_%Y%m%d_%H%M%S.png")
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
