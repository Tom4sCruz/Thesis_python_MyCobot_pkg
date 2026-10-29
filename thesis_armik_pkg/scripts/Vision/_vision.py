"""
VISION helpers shared by make_markers.py, calibrate_camera.py,
calibrate_base.py and reach_cube.py.

Marker layout (DICT_4X4_50, printed by make_markers.py):
  IDs 0-5   cube faces, 3.0 cm black square on a 3.4 cm white tile
  ID  10    BASE   -- flat on the table next to the robot, 8 cm
  ID  11    TARGET -- flat on the table where the cube goes, 8 cm

Frames:
  camera  OpenCV camera frame (x right, y down, z forward), cm
  marker  ArUco marker frame: origin at the marker centre, x to the printed
          marker's right, y to its top, z out of the paper
  robot   armik's base frame, cm (what Arm.send_coords() takes)

The camera -> robot transform comes from the BASE marker, whose pose in the
robot frame is BASE_MARKER_POSE (a ruler-measured guess) refined by the
teach-pairs correction in base_calib.json (calibrate_base.py).

The cube's face IDs and how each tile is rotated when glued do not matter:
the cube centre is (marker origin - side/2 * marker normal) from ANY face, and
the gripper only needs the cube's yaw modulo 90 deg, which any face gives.

Units: cm and degrees throughout. Deliberately outside the armik package so
armik stays OpenCV-free.
"""

from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

ARUCO_DICT = "DICT_4X4_50"
CUBE_FACE_IDS = (0, 1, 2, 3, 4, 5)
BASE_ID = 10
TARGET_ID = 11

CUBE_SIDE_CM = 3.5               # edge of the real cube
CUBE_MARKER_CM = 2.85             # black square of a cube-face marker -- MEASURE the print
CUBE_TILE_CM = 3.4               # white tile it is printed on (cut to this, glue centred)
BIG_MARKER_CM = 7.6              # black square of the BASE / TARGET markers -- MEASURE the print

# Pose of the BASE marker's centre in the robot frame: (x, y, z cm, yaw deg).
# yaw = angle from robot +x to the marker's x axis (its printed "right"),
# counter-clockwise seen from above. make_markers.py prints "robot +X" above
# the BASE marker: lay it with that arrow pointing along robot +x and the
# marker x axis points along robot -y, i.e. yaw = -90.
# z is the TABLE height in the robot frame (the marker lies on the table);
# config.GROUND_Z_MM puts the floor at -6 cm. Measure x/y from the J1 axis to
# the marker centre; calibrate_base.py then corrects whatever is left.
BASE_MARKER_POSE = (-12.0, 0.0, -6.0, -90.0)

BASE_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_calib.json")
CAMERA_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calib.npz")

CAMERA_SOURCE = "realsense"      # "realsense" (pyrealsense2) or "v4l2" (cv2.VideoCapture)
CAMERA_INDEX = 0                 # v4l2 only: /dev/videoN (a RealSense exposes several nodes)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Two-solution pose ambiguity (IPPE): of the two candidate poses of a marker,
# keep the one whose normal is closest to a robot axis (cube faces are flat or
# upright), as long as its reprojection error is at most this times the best one.
IPPE_ERROR_RATIO = 3.0
CUBE_FACE_MAX_TILT_DEG = 25.0    # a face whose normal is this far from every robot axis
                                 # (vertical or horizontal) is rejected as a bad pose


# =============================================================================
# CAMERA
# =============================================================================

class _RealSenseCamera:
    """Colour stream of a RealSense through pyrealsense2 (any D4xx model)."""

    def __init__(self, width, height, fps):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise SystemExit(
                "pyrealsense2 is not installed. On the Jetson build librealsense with "
                "-DBUILD_PYTHON_BINDINGS=ON, or use --source v4l2 --camera N")
        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipe.start(cfg)
        self._intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.name = f"RealSense colour {width}x{height}@{fps}"

    def read(self):
        frames = self._pipe.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data()).copy()

    def intrinsics(self):
        """Factory calibration: (K 3x3, dist (5,), (w, h))."""
        i = self._intr
        K = np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]])
        return K, np.array(i.coeffs, dtype=float), (i.width, i.height)

    def release(self):
        self._pipe.stop()


class _V4L2Camera:
    def __init__(self, index, width, height, fps):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open camera /dev/video{index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.name = f"/dev/video{index}"

    def read(self):
        return self._cap.read()

    def intrinsics(self):
        return None

    def release(self):
        self._cap.release()


def open_camera(source=CAMERA_SOURCE, index=CAMERA_INDEX,
                width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
    """A camera object with .read() -> (ok, bgr), .intrinsics() and .release()."""
    if source == "realsense":
        return _RealSenseCamera(width, height, fps)
    if source == "v4l2":
        return _V4L2Camera(index, width, height, fps)
    raise SystemExit(f"unknown camera source {source!r} (realsense or v4l2)")


def add_camera_args(ap):
    """The camera flags every vision script shares."""
    ap.add_argument("--source", choices=("realsense", "v4l2"), default=CAMERA_SOURCE,
                    help="realsense (pyrealsense2) or v4l2 (cv2.VideoCapture)")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX, help="v4l2 only: /dev/videoN")
    ap.add_argument("--width", type=int, default=CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=CAMERA_HEIGHT)


def camera_model(cam, width, height, calib_path=None):
    """(K, dist, description). Priority: --calib file, the RealSense factory
    intrinsics, then a rough pinhole guess (fine for eyeballing, not for grasping)."""
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
        return K, dist, f"calibrated ({calib_path})"
    intr = cam.intrinsics()
    if intr is not None:
        K, dist, _ = intr
        return K, dist, "RealSense factory intrinsics"
    print("WARNING: no --calib and no factory intrinsics -- guessing the camera model; "
          "positions will be off. Run calibrate_camera.py.", file=sys.stderr)
    f = float(width)
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return K, np.zeros(5), "approximate intrinsics (uncalibrated)"


# =============================================================================
# DETECTION + POSES
# =============================================================================

def make_detector(dict_name=ARUCO_DICT):
    if not hasattr(cv2.aruco, dict_name):
        raise SystemExit(f"Unknown ArUco dictionary {dict_name!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect(detector, frame):
    """{marker_id: (4, 2) corners} -- only the first sighting of each ID."""
    corners, ids, _ = detector.detectMarkers(frame)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out.setdefault(int(i), c.reshape(4, 2).astype(np.float64))
    return out


def marker_object_points(length):
    """Marker corners in the marker frame, in the detector's order: top-left,
    top-right, bottom-right, bottom-left (what SOLVEPNP_IPPE_SQUARE expects)."""
    h = length / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def marker_pose_candidates(corners, length, K, dist):
    """Both IPPE solutions as [(T_cam_marker 4x4, reprojection error px)], best first."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        marker_object_points(length), corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = np.asarray(errs, dtype=float).ravel() if errs is not None else np.zeros(n)
    out = [(rt_to_T(r, t), float(e)) for r, t, e in zip(rvecs, tvecs, errs)]
    return sorted(out, key=lambda p: p[1])


def marker_pose(corners, length, K, dist):
    """The lower-error IPPE solution, T_cam_marker (4x4)."""
    return marker_pose_candidates(corners, length, K, dist)[0][0]


def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    T[:3, 3] = np.asarray(tvec, dtype=float).ravel()
    return T


def T_to_rt(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec, T[:3, 3].reshape(3, 1)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mean_pose(Ts):
    """Average of several 4x4 poses: mean translation, rotation = the
    nearest rotation to the mean matrix (SVD)."""
    Ts = np.asarray(Ts)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.median(Ts[:, :3, 3], axis=0)
    return T


# =============================================================================
# CAMERA -> ROBOT
# =============================================================================

def load_base_calib(path=BASE_CALIB_PATH):
    """Teach-pairs correction from calibrate_base.py, or the identity if none:
    {"yaw_deg", "dx_cm", "dy_cm", "dz_cm"} applied AFTER the nominal transform."""
    if path and os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        fitted_for = data.get("base_marker_pose")
        if fitted_for is not None and not np.allclose(fitted_for, BASE_MARKER_POSE):
            print(f"WARNING: {path} was fitted for BASE_MARKER_POSE {fitted_for}, now "
                  f"{list(BASE_MARKER_POSE)} -- re-run calibrate_base.py", file=sys.stderr)
        return {k: float(data.get(k, 0.0)) for k in ("yaw_deg", "dx_cm", "dy_cm", "dz_cm")}
    return {"yaw_deg": 0.0, "dx_cm": 0.0, "dy_cm": 0.0, "dz_cm": 0.0}


def correction_T(calib):
    T = np.eye(4)
    T[:3, :3] = rot_z(calib["yaw_deg"])
    T[:3, 3] = [calib["dx_cm"], calib["dy_cm"], calib["dz_cm"]]
    return T


def robot_T_base_marker(pose=BASE_MARKER_POSE):
    x, y, z, yaw = pose
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw)
    T[:3, 3] = [x, y, z]
    return T


def robot_from_camera(T_cam_base, calib=None, pose=BASE_MARKER_POSE):
    """T_robot_cam: maps camera-frame points (cm) into the robot frame."""
    T = robot_T_base_marker(pose) @ np.linalg.inv(T_cam_base)
    if calib is not None:
        T = correction_T(calib) @ T
    return T


def table_z(calib=None, pose=BASE_MARKER_POSE):
    """Table height in the robot frame (the base marker lies on it)."""
    return pose[2] + (calib["dz_cm"] if calib else 0.0)


def axis_alignment(R_robot_marker):
    """How close the marker normal is to a robot axis: 1 = exactly vertical
    or exactly horizontal-and-along-the-cube, falls off with tilt. Returns
    the tilt in degrees from the nearest of {vertical, horizontal}."""
    nz = abs(float(R_robot_marker[2, 2]))           # |cos| of angle to vertical
    tilt_vertical = math.degrees(math.acos(min(1.0, nz)))
    return min(tilt_vertical, 90.0 - tilt_vertical)


def pick_aligned_pose(candidates, T_robot_cam):
    """Of the IPPE candidates, the one whose normal is closest to vertical or
    horizontal in the robot frame, among those with an acceptable error.
    Returns (T_cam_marker, tilt_deg)."""
    best_err = candidates[0][1]
    ok = [T for T, e in candidates if e <= max(best_err * IPPE_ERROR_RATIO, best_err + 0.5)]
    scored = [(axis_alignment((T_robot_cam @ T)[:3, :3]), T) for T in ok]
    tilt, T = min(scored, key=lambda s: s[0])
    return T, tilt


def cube_from_faces(markers, K, dist, T_robot_cam, side=CUBE_SIDE_CM,
                    marker_len=CUBE_MARKER_CM, face_ids=CUBE_FACE_IDS, table_z=None):
    """Fuse every visible cube face into one estimate.

    table_z: if given, the cube is assumed to rest on the table. Each face's
    marker centre is then slid along its camera ray onto the height it must
    have (table + side for the top face, table + side/2 for a side face). The
    bearing from a small marker is accurate but its RANGE is not (a 3 px
    corner error on a 45 px marker is ~5 cm of range at 60 cm), and that
    error runs along the ray -- so x/y need the correction too, not just z.

    Returns dict(center=(3,) robot cm, yaw=deg in [-45, 45), faces=[ids],
    face_poses=[(id, T_cam_marker)], raw_z=cube z before the table
    correction) or None if no usable face is visible.
    yaw = direction of a cube side's normal in the robot xy plane, mod 90.
    """
    T_cam_robot = np.linalg.inv(T_robot_cam)
    cam = T_robot_cam[:3, 3]
    centers, weights, yaw_vecs, used, poses, raw_z = [], [], [], [], [], []
    for mid in face_ids:
        if mid not in markers:
            continue
        cands = marker_pose_candidates(markers[mid], marker_len, K, dist)
        T_cm, tilt = pick_aligned_pose(cands, T_robot_cam)
        if tilt > CUBE_FACE_MAX_TILT_DEG:
            continue
        T_rm = T_robot_cam @ T_cm
        R = T_rm[:3, :3]
        p = T_rm[:3, 3]
        raw_z.append(float(p[2] - (side / 2.0) * R[2, 2]))
        if table_z is not None:
            z_face = table_z + (side if abs(R[2, 2]) > 0.7 else side / 2.0)
            ray = p - cam
            if abs(ray[2]) > 1e-3 * np.linalg.norm(ray):
                p = cam + ray * ((z_face - cam[2]) / ray[2])
        center = p - (side / 2.0) * R[:, 2]
        # Weight: how squarely the face looks at the camera (cos of view angle).
        to_cam = T_cam_robot[:3, 3] - T_rm[:3, 3]
        w = max(float(np.dot(R[:, 2], to_cam / np.linalg.norm(to_cam))), 0.05)
        # Yaw from whichever marker axis is most horizontal (a cube side's direction).
        axis = R[:, int(np.argmin(np.abs(R[2, :])))]
        a = math.atan2(axis[1], axis[0])
        centers.append(center)
        weights.append(w)
        yaw_vecs.append(w * np.array([math.cos(4 * a), math.sin(4 * a)]))
        used.append(mid)
        poses.append((mid, T_cm))
    if not used:
        return None
    w = np.array(weights)
    center = (np.array(centers) * w[:, None]).sum(axis=0) / w.sum()
    v = np.sum(yaw_vecs, axis=0)
    yaw = math.degrees(math.atan2(v[1], v[0])) / 4.0     # in (-45, 45]
    return {"center": center, "yaw": wrap90(yaw), "faces": used, "face_poses": poses,
            "raw_z": float(np.dot(raw_z, w) / w.sum())}


def wrap90(deg):
    """Fold an angle into [-45, 45) -- a cube looks the same every 90 deg."""
    return (deg + 45.0) % 90.0 - 45.0


# =============================================================================
# DRAWING
# =============================================================================

def put_text(img, text, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def project_robot_points(pts_robot, T_robot_cam, K, dist):
    """(N, 3) robot-frame cm -> (N, 2) pixels."""
    T_cam_robot = np.linalg.inv(T_robot_cam)
    rvec, tvec = T_to_rt(T_cam_robot)
    px, _ = cv2.projectPoints(np.asarray(pts_robot, dtype=np.float64).reshape(-1, 3),
                              rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def _pt(p):
    return tuple(int(v) for v in np.round(p))


def draw_robot_axes(img, T_robot_cam, K, dist, length=8.0):
    """Robot base frame at the origin: x red, y green, z blue."""
    pts = project_robot_points([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]],
                               T_robot_cam, K, dist)
    for end, color in zip(pts[1:], ((0, 0, 255), (0, 200, 0), (255, 80, 0))):
        cv2.arrowedLine(img, _pt(pts[0]), _pt(end), color, 2, cv2.LINE_AA, tipLength=0.15)
    put_text(img, "robot", _pt(pts[0] + 6))


def draw_cube_box(img, center, yaw, side, T_robot_cam, K, dist, color=(0, 200, 255)):
    """Wireframe of the fused cube estimate, in the robot frame."""
    h = side / 2.0
    R = rot_z(yaw)
    corners = np.array([[sx * h, sy * h, sz * h] for sz in (-1, 1)
                        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    pts = project_robot_points(corners @ R.T + center, T_robot_cam, K, dist)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)):
        cv2.line(img, _pt(pts[a]), _pt(pts[b]), color, 2, cv2.LINE_AA)


def draw_point(img, p_robot, T_robot_cam, K, dist, color, label=None, radius=6):
    px = project_robot_points([p_robot], T_robot_cam, K, dist)[0]
    cv2.circle(img, _pt(px), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, _pt(px), radius, (0, 0, 0), 1, cv2.LINE_AA)
    if label:
        put_text(img, label, _pt(px + [radius + 4, -radius]), color)
