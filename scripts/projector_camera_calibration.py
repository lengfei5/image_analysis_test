"""
Basler camera ROI adjustment + projector-camera homography calibration.

Keys (in Control Window):
  Arrow keys .... move ROI (OffsetX/OffsetY)
  w/s ........... ROI width  +/- (step 32)
  h/g ........... ROI height +/- (step 32)
  c ............. run homography calibration (projects ArUco markers)
  r ............. reset homography
  ESC/q ......... quit

"""
import sys
import time
import numpy as np
import cv2
from screeninfo import get_monitors
from pypylon.pylon import (InstantCamera, TlFactory,
                           GrabStrategy_LatestImageOnly,
                           TimeoutHandling_ThrowException)

ROI_STEP = 32          # ROI resize step
MOVE_STEP = 16         # ROI move step
MARKER_MARGIN = 80     # marker inset from projector edges (px)
MARKER_SIZE = 200      # projected marker size (px)

# ---------------------------------------------------------------- camera ROI
def snap(v, mult):
    return (v // mult) * mult

def set_roi(cam, w, h, ox, oy):
    """Safely apply ROI (respecting increments and limits)."""
    wmax, hmax = cam.WidthMax.Value, cam.HeightMax.Value
    w  = max(64, min(snap(w, 4), wmax))
    h  = max(64, min(snap(h, 2), hmax))
    ox = max(0, min(snap(ox, 4), wmax - w))
    oy = max(0, min(snap(oy, 2), hmax - h))
    cam.StopGrabbing()
    cam.OffsetX.SetValue(0); cam.OffsetY.SetValue(0)
    cam.Width.SetValue(w);   cam.Height.SetValue(h)
    cam.OffsetX.SetValue(ox); cam.OffsetY.SetValue(oy)
    cam.StartGrabbing(GrabStrategy_LatestImageOnly)
    return w, h, ox, oy

def grab_frame(cam):
    res = cam.RetrieveResult(2000, TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        res.Release()
        return None
    frame = res.Array.copy()
    res.Release()
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

# ------------------------------------------------------------- calibration

def make_marker_canvas(proj_w, proj_h):
    """White canvas with 4 ArUco markers at known projector coordinates."""
    aruco = cv2.aruco
    dic = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    canvas = np.full((proj_h, proj_w), 255, np.uint8)
    m, s = MARKER_MARGIN, MARKER_SIZE
    positions = [(m, m), (proj_w - m - s, m),
                 (proj_w - m - s, proj_h - m - s), (m, proj_h - m - s)]
    proj_pts = []
    for i, (x, y) in enumerate(positions):
        img = aruco.generateImageMarker(dic, i, s)
        canvas[y:y + s, x:x + s] = img
        # marker corners in projector coords (TL, TR, BR, BL)
        proj_pts.append([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])
    return canvas, np.array(proj_pts, np.float32), dic

def calibrate_homography(cam, projwin, proj_w, proj_h):
    """Project ArUco markers, detect them with the camera, compute H (camera->projector)."""
    canvas, proj_pts, dic = make_marker_canvas(proj_w, proj_h)
    cv2.imshow(projwin, canvas)
    cv2.waitKey(500)                      # let projector display + exposure settle
    time.sleep(0.3)

    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    for _ in range(10):                   # try a few frames
        frame = grab_frame(cam)
        if frame is None:
            continue
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) >= 4:
            cam_pts, prj_pts = [], []
            for c, i in zip(corners, ids.flatten()):
                if i < 4:
                    cam_pts.append(c.reshape(4, 2))
                    prj_pts.append(proj_pts[i])
            if len(cam_pts) == 4:
                cam_pts = np.concatenate(cam_pts)
                prj_pts = np.concatenate(prj_pts)
                H, mask = cv2.findHomography(cam_pts, prj_pts, cv2.RANSAC, 3.0)
                err = np.linalg.norm(
                    cv2.perspectiveTransform(cam_pts.reshape(-1, 1, 2), H).reshape(-1, 2) - prj_pts,
                    axis=1).mean()
                print(f'Calibration OK, mean reprojection error: {err:.2f} px')
                return H
        cv2.waitKey(100)
    print('Calibration FAILED: could not detect all 4 markers.')
    return None

# -------------------------------------------------------------------- main

def main():
    # camera
    try:
        cam = InstantCamera(TlFactory.GetInstance().CreateFirstDevice())
        cam.Open()
    except Exception as e:
        print(f'Cannot open camera: {e}')
        return 1
    # sensible defaults: full sensor, no auto features drifting during calib
    try:
        cam.ExposureAuto.SetValue('Off')
        cam.GainAuto.SetValue('Off')
    except Exception:
        pass
    w, h = cam.WidthMax.Value, cam.HeightMax.Value
    ox = oy = 0
    w, h, ox, oy = set_roi(cam, w, h, ox, oy)

    # windows
    monitors = get_monitors()
    ctrlwin = 'Control Window'
    cv2.namedWindow(ctrlwin, cv2.WINDOW_NORMAL)
    size = monitors[0].height * 8 // 10
    cv2.resizeWindow(ctrlwin, size, size)
    cv2.moveWindow(ctrlwin, monitors[0].x + 50, monitors[0].y + 50)

    have_proj = len(monitors) > 1
    projwin, proj_w, proj_h = None, 0, 0
    if have_proj:
        projwin = 'Projection Window'
        proj_w, proj_h = monitors[1].width, monitors[1].height
        cv2.namedWindow(projwin, cv2.WINDOW_NORMAL)
        cv2.moveWindow(projwin, monitors[1].x, monitors[1].y)
        cv2.resizeWindow(projwin, proj_w, proj_h)
        cv2.setWindowProperty(projwin, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        print('No second screen detected - projector features disabled.')

    H = None  # camera -> projector homography

    while cam.IsGrabbing():
        frame = grab_frame(cam)
        if frame is None:
            break

        # control window: show frame + status text
        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        status = f'ROI {w}x{h} @ ({ox},{oy}) | H: {"calibrated" if H is not None else "none"}'
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow(ctrlwin, disp)

        # projector window: geometrically registered view
        if have_proj:
            if H is not None:
                out = cv2.warpPerspective(frame, H, (proj_w, proj_h))
            else:
                s = max(frame.shape)
                sq = cv2.resize(frame, (s, s))
                pw = max(0, round((s * proj_w / proj_h - s) / 2))
                out = np.pad(sq, ((0, 0), (pw, pw)), constant_values=0)
            cv2.imshow(projwin, out)

        key = cv2.waitKeyEx(1)
        if key in (27, ord('q')):
            break
        elif key == ord('c') and have_proj:
            Hnew = calibrate_homography(cam, projwin, proj_w, proj_h)
            if Hnew is not None:
                H = Hnew
        elif key == ord('r'):
            H = None
        elif key == ord('w'):
            w, h, ox, oy = set_roi(cam, w + ROI_STEP, h, ox, oy); H = None
        elif key == ord('s'):
            w, h, ox, oy = set_roi(cam, w - ROI_STEP, h, ox, oy); H = None
        elif key == ord('h'):
            w, h, ox, oy = set_roi(cam, w, h + ROI_STEP, ox, oy); H = None
        elif key == ord('g'):
            w, h, ox, oy = set_roi(cam, w, h - ROI_STEP, ox, oy); H = None
        elif key == 2424832:  # left arrow
            w, h, ox, oy = set_roi(cam, w, h, ox - MOVE_STEP, oy); H = None
        elif key == 2555904:  # right arrow
            w, h, ox, oy = set_roi(cam, w, h, ox + MOVE_STEP, oy); H = None
        elif key == 2490368:  # up arrow
            w, h, ox, oy = set_roi(cam, w, h, ox, oy - MOVE_STEP); H = None
        elif key == 2621440:  # down arrow
            w, h, ox, oy = set_roi(cam, w, h, ox, oy + MOVE_STEP); H = None

    if H is not None:
        np.save('homography.npy', H)
        print('Homography saved to homography.npy')

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main())
