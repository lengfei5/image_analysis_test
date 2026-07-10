"""
Basler camera ROI adjustment (PyQt5 GUI) + projector-camera homography calibration.

GUI:
  Sliders/edits ... ROI width/height/offset (with "zoom together")
  Calibrate ....... project ArUco markers, compute camera->projector homography
  Reset H ......... discard homography
  Done ............ quit (saves homography.npy if calibrated)

Control Window keys: ESC or q also quits.

"""
import os
import sys
import time
import numpy as np
import cv2
from screeninfo import get_monitors

# Detects monitors → identifies the projector as the second display.
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QDesktopWidget,
                             QWidget, QLabel, QLineEdit, QSlider, QPushButton, QCheckBox)

# Basler API: camera object, device factory, "keep only newest frame" grab strategy (low latency), and exception-on-timeout mode.
from pypylon.pylon import (InstantCamera, TlFactory,
                           GrabStrategy_LatestImageOnly, TimeoutHandling_ThrowException

# Tries to import your project helpers (popups + camera config).
# If utils.py is missing, defines simple fallbacks that just print to console, so the script still runs standalone.
try:
    from utils import show_info, show_error, setup_camera
except ImportError:  # fallback if utils.py is unavailable
    def show_info(msg): print(f'INFO: {msg}')
    def show_error(msg): print(f'ERROR: {msg}')
    def setup_camera(cam): pass

# Markers are placed 80 px in from the projector's corners.
MARKER_MARGIN = 80     # marker inset from projector edges (px)
# Each ArUco marker is 200×200 px.
MARKER_SIZE = 200      # projected marker size (px)

HOMOGRAPHY_FILE = 'homography.npy'


# ---------- camera ROI ----------
# Rounds v down to the nearest multiple of mult (e.g., snap(103, 4) → 100). Basler ROI values must be multiples of specific increments.
def snap(v, mult):
    return (v // mult) * mult

# Reads the sensor's maximum dimensions.
def set_roi(cam, w, h, ox, oy):
    """Safely apply ROI (respecting increments and limits)."""
    wmax, hmax = cam.WidthMax.Value, cam.HeightMax.Value
    w  = max(64, min(snap(w, 4), wmax)) # Clamps width to [64, wmax] in multiples of 4; height to [64, hmax] in multiples of 2.
    h  = max(64, min(snap(h, 2), hmax))
    ox = max(0, min(snap(ox, 4), wmax - w)) # Clamps offsets so offset + size never exceeds the sensor.
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
    frame = cv2.flip(frame, -1)  # 180-degree rotation to match physical mounting
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


# ---------- calibration ----------
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
        proj_pts.append([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])
    return canvas, np.array(proj_pts, np.float32), dic

def calibrate_homography(cam, projwin, proj_w, proj_h):
    """Project ArUco markers, detect them with the camera, compute H (camera->projector)."""
    canvas, proj_pts, dic = make_marker_canvas(proj_w, proj_h)
    cv2.imshow(projwin, canvas)
    cv2.waitKey(500)  # let projector display + exposure settle
    time.sleep(0.3)

    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    for _ in range(10):
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
                H, _ = cv2.findHomography(cam_pts, prj_pts, cv2.RANSAC, 3.0)
                err = np.linalg.norm(
                    cv2.perspectiveTransform(cam_pts.reshape(-1, 1, 2), H).reshape(-1, 2) - prj_pts,
                    axis=1).mean()
                show_info(f'Calibration OK, mean reprojection error: {err:.2f} px')
                return H
        cv2.waitKey(100)
    show_error('Calibration FAILED: could not detect all 4 markers.\n'
               'Check projector brightness / camera exposure / field of view.')
    return None


# ---------- Qt GUI ----------

class MultiplesValidator(QValidator):
    def __init__(self, multiple, min_, max_):
        super().__init__()
        self.multiple, self.min, self.max = multiple, min_, max_

    def validate(self, input, pos):
        if input.isdigit():
            value = int(input)
            if self.min <= value <= self.max and value % self.multiple == 0:
                return (QValidator.Acceptable, input, pos)
            return (QValidator.Intermediate, input, pos)
        return (QValidator.Invalid, input, pos)


class ParameterControl(QWidget):
    """Slider GUI for camera ROI + calibration buttons.

    Communicates with the main loop via flags:
      self.request_calibration, self.request_reset_h, self.roi_changed
    """
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.wmax = camera.WidthMax.Value
        self.hmax = camera.HeightMax.Value
        self.w = camera.Width.GetValue()
        self.h = camera.Height.GetValue()
        self.ox = camera.OffsetX.GetValue()
        self.oy = camera.OffsetY.GetValue()

        self.request_calibration = False
        self.request_reset_h = False
        self.roi_changed = False
        self._updating = False  # guard against widget signal loops

        # zoom together
        self.checkbox_zoomxy = QCheckBox()
        self.checkbox_zoomxy.setChecked(False)

        def make_row(label, minimum, maximum, step, value, callback):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(minimum); slider.setMaximum(maximum)
            slider.setSingleStep(step); slider.setValue(value)
            slider.valueChanged.connect(lambda: None if self._updating else callback(slider.value()))
            edit = QLineEdit(str(value))
            edit.setValidator(MultiplesValidator(step, minimum, maximum))
            edit.setFixedWidth(50)
            edit.editingFinished.connect(lambda: None if self._updating else callback(int(edit.text())))
            return QLabel(label), slider, edit

        self.label_w, self.slider_w, self.edit_w = make_row('Zoom X', 64, self.wmax, 4, self.w, self.set_width)
        self.label_h, self.slider_h, self.edit_h = make_row('Zoom Y', 64, self.hmax, 2, self.h, self.set_height)
        self.label_ox, self.slider_ox, self.edit_ox = make_row('Offset X', 0, self.wmax - self.w, 4, self.ox, self.set_offsetx)
        self.label_oy, self.slider_oy, self.edit_oy = make_row('Offset Y', 0, self.hmax - self.h, 2, self.oy, self.set_offsety)

        self.button_calib = QPushButton('Calibrate (project markers)')
        self.button_calib.clicked.connect(self._on_calibrate)
        self.button_reset = QPushButton('Reset homography')
        self.button_reset.clicked.connect(self._on_reset)
        self.button_done = QPushButton('Done')
        self.button_done.clicked.connect(self.close)

        layout_main = QVBoxLayout()
        row = QHBoxLayout(); row.setAlignment(Qt.AlignLeft)
        row.addWidget(QLabel('Zoom together')); row.addWidget(self.checkbox_zoomxy)
        layout_main.addLayout(row)
        for lab, sld, ed in [(self.label_w, self.slider_w, self.edit_w),
                             (self.label_h, self.slider_h, self.edit_h),
                             (self.label_ox, self.slider_ox, self.edit_ox),
                             (self.label_oy, self.slider_oy, self.edit_oy)]:
            row = QHBoxLayout(); row.setAlignment(Qt.AlignLeft)
            row.addWidget(lab); row.addWidget(sld); row.addWidget(ed)
            layout_main.addLayout(row)
        for b in [self.button_calib, self.button_reset, self.button_done]:
            layout_main.addWidget(b)
        self.setLayout(layout_main)

        self.setWindowTitle('Adjust Camera')
        self.setMinimumWidth(300)
        self.adjustSize()
        _, _, sw, sh = QDesktopWidget().screenGeometry().getRect()
        _, _, w, h = self.geometry().getRect()
        self.move(sw * 3 // 4 - w // 2, sh // 2 - h * 3 // 4)

    # --- ROI setters: all funnel through _apply_roi (safe clamping) ---

    def _apply_roi(self, w, h, ox, oy):
        self.w, self.h, self.ox, self.oy = set_roi(self.camera, w, h, ox, oy)
        self.roi_changed = True  # invalidates homography in main loop
        self._sync_widgets()

    def _sync_widgets(self):
        self._updating = True
        self.slider_w.setValue(self.w);  self.edit_w.setText(str(self.w))
        self.slider_h.setValue(self.h);  self.edit_h.setText(str(self.h))
        self.slider_ox.setMaximum(self.wmax - self.w)
        self.edit_ox.setValidator(MultiplesValidator(4, 0, self.wmax - self.w))
        self.slider_ox.setValue(self.ox); self.edit_ox.setText(str(self.ox))
        self.slider_oy.setMaximum(self.hmax - self.h)
        self.edit_oy.setValidator(MultiplesValidator(2, 0, self.hmax - self.h))
        self.slider_oy.setValue(self.oy); self.edit_oy.setText(str(self.oy))
        self._updating = False

    def set_width(self, w_new):
        h_new = w_new if self.checkbox_zoomxy.isChecked() else self.h
        self._apply_roi(w_new, h_new, self.ox, self.oy)

    def set_height(self, h_new):
        w_new = h_new if self.checkbox_zoomxy.isChecked() else self.w
        self._apply_roi(w_new, h_new, self.ox, self.oy)

    def set_offsetx(self, ox_new):
        self._apply_roi(self.w, self.h, ox_new, self.oy)

    def set_offsety(self, oy_new):
        self._apply_roi(self.w, self.h, self.ox, oy_new)

    # --- buttons ---
    def _on_calibrate(self):
        self.request_calibration = True

    def _on_reset(self):
        self.request_reset_h = True

    def closeEvent(self, event):
        QApplication.exit(0)


# ---------- main ----------

def main():
    # camera
    try:
        cam = InstantCamera(TlFactory.GetInstance().CreateFirstDevice())
    except Exception:
        show_error('Cannot access the camera. Make sure all other software that accesses it is closed.')
        return -1
    cam.Open()
    if not cam.IsOpen():
        show_error('Cannot find camera.')
        return -1
    setup_camera(cam)
    try:
        cam.ExposureAuto.SetValue('Off')
        cam.GainAuto.SetValue('Off')
    except Exception:
        pass
    set_roi(cam, cam.WidthMax.Value, cam.HeightMax.Value, 0, 0)
    cam.StopGrabbing()  # set_roi started grabbing; restart cleanly below

    # windows
    monitors = get_monitors()
    ctrlwin = 'Control Window'
    cv2.namedWindow(ctrlwin, cv2.WINDOW_NORMAL)
    size = monitors[0].height * 8 // 10
    cv2.resizeWindow(ctrlwin, size, size)
    cv2.moveWindow(ctrlwin, monitors[0].x + monitors[0].height // 10,
                   monitors[0].y + monitors[0].height // 10)

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
        show_info('No second screen detected - projector features disabled.')

    # load existing homography if available
    H = None
    if os.path.isfile(HOMOGRAPHY_FILE):
        H = np.load(HOMOGRAPHY_FILE)
        print(f'Loaded existing homography from {HOMOGRAPHY_FILE}')

    # Qt dialog
    app = QApplication(sys.argv)
    controller = ParameterControl(cam)
    controller.show()

    cam.StartGrabbing(GrabStrategy_LatestImageOnly)
    while (cam.IsGrabbing() and controller.isVisible()
           and cv2.getWindowProperty(ctrlwin, cv2.WND_PROP_VISIBLE)):

        # handle GUI requests
        if controller.roi_changed:
            controller.roi_changed = False
            H = None  # ROI change invalidates homography
        if controller.request_reset_h:
            controller.request_reset_h = False
            H = None
        if controller.request_calibration:
            controller.request_calibration = False
            if have_proj:
                Hnew = calibrate_homography(cam, projwin, proj_w, proj_h)
                if Hnew is not None:
                    H = Hnew
            else:
                show_error('No projector detected - cannot calibrate.')

        frame = grab_frame(cam)
        if frame is None:
            break

        # control window: frame + status text
        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        status = (f'ROI {controller.w}x{controller.h} @ ({controller.ox},{controller.oy}) | '
                  f'H: {"calibrated" if H is not None else "none"}')
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(ctrlwin, disp)

        # projector window
        if have_proj:
            if H is not None:
                out = cv2.warpPerspective(frame, H, (proj_w, proj_h))
            else:
                s = max(frame.shape)
                sq = cv2.resize(frame, (s, s))
                pw = max(0, round((s * proj_w / proj_h - s) / 2))
                out = np.pad(sq, ((0, 0), (pw, pw)), constant_values=0)
            cv2.imshow(projwin, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    if H is not None:
        np.save(HOMOGRAPHY_FILE, H)
        print(f'Homography saved to {HOMOGRAPHY_FILE}')

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows()
    controller.close()
    app.quit()
    return 0


if __name__ == '__main__':
    print('PROGRAM STARTED')
    sys.exit(main())
