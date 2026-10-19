#!/usr/bin/env python3
"""
VISION: camera calibration (chessboard) -> camera_calib.npz
===========================================================

Estimates the camera's intrinsics (focal lengths, principal point) and lens
distortion from views of a printed chessboard, and saves them for
aruco_cube.py --calib. Without this, aruco_cube.py guesses the intrinsics,
which is fine for eyeballing the overlay but not for measuring positions.

How to calibrate:
  1. python3 scripts/Vision/calibrate_camera.py --print-board
     Print chessboard_9x6.png at 100 % scale (no "fit to page"), measure one
     square with a ruler and tape/glue the sheet FLAT (cardboard, a book).
  2. python3 scripts/Vision/calibrate_camera.py --square-cm <measured>
     Hold the board in front of the camera. When the corners are drawn in
     colour the board is found -- press SPACE to keep that view. Collect ~20
     views: near and far, the board at all four corners and edges of the image
     (not just the centre), tilted ~30-45 deg in different directions. Keep it
     still and sharp (no motion blur).
  3. Press C -> calibrates and writes camera_calib.npz. RMS reprojection
     error: < 0.5 px good, < 1 px OK, more -> recapture (blurry/bent views).
  4. python3 scripts/Vision/aruco_cube.py --calib camera_calib.npz
The calibration is only valid for this camera at this resolution.

Keys (live): SPACE keep view, U drop last view, C calibrate + save, Q/Esc quit.

Saved .npz keys: camera_matrix (3x3), dist_coeffs, image_size (w, h), rms.
Units: --square-cm sets the board units (cm), which only affects the
per-view translations, not the intrinsics.
"""

import argparse
import sys

import cv2
import numpy as np


# =============================================================================
# CONSTANTS  (each one can be overridden from the command line)
# =============================================================================

BOARD_COLS = 9                   # INNER corners per row (squares per row - 1)
BOARD_ROWS = 6                   # INNER corners per column
SQUARE_CM = 2.5                  # printed square side -- measure after printing
MIN_VIEWS = 15                   # views needed before C is accepted
CAMERA_INDEX = 0                 # /dev/videoN
OUT_PATH = "camera_calib.npz"

DETECT_WIDTH_PX = 640            # corners are searched on a copy this wide (speed), then
                                 # refined with cornerSubPix on the full-resolution image
BOARD_SQUARE_PX = 200            # --print-board: pixels per square (~2.5 cm at 200 dpi)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


# =============================================================================
# HELPERS
# =============================================================================

def board_object_points(cols, rows, square):
    """Chessboard inner corners in the board frame (z = 0), row by row."""
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def find_corners(gray, cols, rows):
    """Inner corners at full resolution, or None. Detects on a downscaled copy for speed."""
    scale = min(1.0, DETECT_WIDTH_PX / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, (cols, rows), flags)
    if not found:
        return None
    corners = (corners / scale).astype(np.float32)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


def calibrate(img_points, image_size, cols, rows, square):
    """Run cv2.calibrateCamera; returns (rms, K, dist, per_view_errors_px)."""
    obj = board_object_points(cols, rows, square)
    obj_points = [obj] * len(img_points)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    errors = []
    for o, i, r, t in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        # Flatten both: corner arrays are (N, 1, 2) in OpenCV 4 but (N, 2) in OpenCV 5.
        diff = proj.reshape(-1, 2) - i.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return rms, K, dist, errors


def save_calibration(path, K, dist, image_size, rms):
    np.savez(path, camera_matrix=K, dist_coeffs=dist,
             image_size=np.array(image_size), rms=rms)


def report(rms, K, dist, errors, image_size):
    print(f"\nCalibrated on {len(errors)} views at {image_size[0]}x{image_size[1]}")
    print(f"  RMS reprojection error: {rms:.3f} px  "
          f"({'good' if rms < 0.5 else 'OK' if rms < 1.0 else 'poor -- recapture'})")
    print(f"  fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}  cx={K[0, 2]:.1f}  cy={K[1, 2]:.1f}")
    print(f"  dist={np.round(dist.ravel(), 4)}")
    worst = sorted(range(len(errors)), key=lambda i: -errors[i])[:3]
    print("  worst views: " + ", ".join(f"#{i + 1} {errors[i]:.2f} px" for i in worst))


def print_board(cols, rows):
    sq = BOARD_SQUARE_PX
    board = np.zeros(((rows + 1) * sq, (cols + 1) * sq), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    board = cv2.copyMakeBorder(board, sq, sq, sq, sq, cv2.BORDER_CONSTANT, value=255)
    path = f"chessboard_{cols}x{rows}.png"
    cv2.imwrite(path, board)
    print(f"Wrote {path} ({cols + 1}x{rows + 1} squares = {cols}x{rows} inner corners). "
          "Print at 100 %, measure one square and pass it as --square-cm.")


def put_text(img, text, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Chessboard camera calibration -> .npz")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX, help="camera index (/dev/videoN)")
    ap.add_argument("--cols", type=int, default=BOARD_COLS, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=BOARD_ROWS, help="inner corners per column")
    ap.add_argument("--square-cm", type=float, default=SQUARE_CM, help="printed square side, cm")
    ap.add_argument("--min-views", type=int, default=MIN_VIEWS, help="views needed to calibrate")
    ap.add_argument("--out", default=OUT_PATH, help="output .npz")
    ap.add_argument("--print-board", action="store_true", help="write a printable chessboard PNG and exit")
    args = ap.parse_args()

    if args.print_board:
        print_board(args.cols, args.rows)
        return 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    window = "calibrate_camera  (SPACE keep, U undo, C calibrate, Q quit)"
    views = []
    image_size = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed", file=sys.stderr)
                return 1
            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, args.cols, args.rows)

            shown = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(shown, (args.cols, args.rows), corners, True)
            status = "board FOUND - SPACE to keep" if corners is not None else "board not found"
            put_text(shown, f"views: {len(views)}/{args.min_views}   {status}", (10, 25),
                     (0, 255, 0) if corners is not None else (0, 0, 255))
            cv2.imshow(window, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Quit without saving.")
                break
            if key == ord(" ") and corners is not None:
                views.append(corners)
                print(f"kept view {len(views)}")
            elif key == ord("u") and views:
                views.pop()
                print(f"dropped last view ({len(views)} left)")
            elif key == ord("c"):
                if len(views) < args.min_views:
                    print(f"need at least {args.min_views} views, have {len(views)}")
                    continue
                rms, K, dist, errors = calibrate(views, image_size, args.cols, args.rows, args.square_cm)
                report(rms, K, dist, errors, image_size)
                save_calibration(args.out, K, dist, image_size, rms)
                print(f"Saved {args.out}  ->  aruco_cube.py --calib {args.out}")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
