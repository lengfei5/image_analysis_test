from sys import argv
from numpy import pad
from screeninfo import get_monitors
from cv2 import namedWindow, moveWindow, resizeWindow, setWindowProperty, getWindowProperty, destroyAllWindows, WINDOW_NORMAL, WINDOW_FULLSCREEN, WND_PROP_FULLSCREEN, WND_PROP_VISIBLE
from cv2 import waitKey, imshow, resize, flip, cvtColor, COLOR_BGR2GRAY
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QDesktopWidget, QWidget, QLabel, QLineEdit, QSlider, QPushButton, QCheckBox
from pypylon.pylon import InstantCamera, TlFactory, GrabStrategy_LatestImageOnly, TimeoutHandling_ThrowException
from utils import show_info, show_error, setup_camera


class MultiplesValidator(QValidator):

	def __init__(self, multiple, min, max):
		super().__init__()
		self.multiple = multiple
		self.min = min
		self.max = max

	def validate(self, input, pos):
		if input.isdigit():
			value = int(input)
			if self.min <= value <= self.max and value % self.multiple == 0:
				return (QValidator.Acceptable, input, pos)
			return (QValidator.Intermediate, input, pos)
		return (QValidator.Invalid, input, pos)


class ParameterControl(QWidget):

	def __init__(self, camera):
		super().__init__()

		self.camera = camera
		self.max_width_orig = camera.WidthMax.Value
		self.max_dim = min(camera.WidthMax.Value, camera.HeightMax.Value)  # force square max image
		init_width = camera.Width.GetValue()
		init_height = camera.Height.GetValue()
		init_offsetx = camera.OffsetX.GetValue()
		init_offsety = camera.OffsetY.GetValue()

		# zoom together
		self.label_zoomxy = QLabel('Zoom together')

		self.checkbox_zoomxy = QCheckBox()
		self.checkbox_zoomxy.setChecked(False)

		# width
		self.label_width = QLabel('Zoom X')

		self.slider_width = QSlider(Qt.Horizontal)
		self.slider_width.setMinimum(4)
		self.slider_width.setMaximum(self.max_dim)
		self.slider_width.setSingleStep(4)
		self.slider_width.setValue(init_width)
		self.slider_width.valueChanged.connect(lambda: self.set_width(self.slider_width.value()))

		self.edit_width = QLineEdit(str(init_width))
		self.edit_width.setValidator(MultiplesValidator(4, 4, self.max_dim))
		self.edit_width.setFixedWidth(50)
		self.edit_width.editingFinished.connect(lambda: self.set_width(int(self.edit_width.text())))

		# height
		self.label_height = QLabel('Zoom Y')

		self.slider_height = QSlider(Qt.Horizontal)
		self.slider_height.setMinimum(1)
		self.slider_height.setMaximum(self.max_dim)
		self.slider_height.setSingleStep(1)
		self.slider_height.setValue(init_height)
		self.slider_height.valueChanged.connect(lambda: self.set_height(self.slider_height.value()))

		self.height_edit = QLineEdit(str(init_height))
		self.height_edit.setValidator(MultiplesValidator(1, 1, self.max_dim))
		self.height_edit.setFixedWidth(50)
		self.height_edit.editingFinished.connect(lambda: self.set_height(int(self.height_edit.text())))

		# offset x
		self.label_offsetx = QLabel('Offset X')

		self.slider_offsetx = QSlider(Qt.Horizontal)
		self.slider_offsetx.setMinimum(0)
		self.slider_offsetx.setMaximum(self.max_width_orig - init_width)
		self.slider_offsetx.setSingleStep(4)
		self.slider_offsetx.setValue(init_offsetx)
		self.slider_offsetx.valueChanged.connect(lambda: self.set_offsetx(self.slider_offsetx.value()))

		self.edit_offsetx = QLineEdit(str(init_offsetx))
		self.edit_offsetx.setValidator(MultiplesValidator(4, 0, self.max_width_orig - init_width))
		self.edit_offsetx.setFixedWidth(50)
		self.edit_offsetx.editingFinished.connect(lambda: self.set_offsetx(int(self.edit_offsetx.text())))

		# offset y
		self.label_offsety = QLabel('Offset Y')

		self.slider_offsety = QSlider(Qt.Horizontal)
		self.slider_offsety.setMinimum(0)
		self.slider_offsety.setMaximum(self.max_dim - init_height)
		self.slider_offsety.setSingleStep(2)
		self.slider_offsety.setValue(init_offsety)
		self.slider_offsety.valueChanged.connect(lambda: self.set_offsety(self.slider_offsety.value()))

		self.edit_offsety = QLineEdit(str(init_offsety))
		self.edit_offsety.setValidator(MultiplesValidator(2, 0, self.max_dim - init_height))
		self.edit_offsety.setFixedWidth(50)
		self.edit_offsety.editingFinished.connect(lambda: self.set_offsety(int(self.edit_offsety.text())))

		# button
		self.button = QPushButton('Done')
		self.button.clicked.connect(self.close)

		# window
		layout_zoomxy = QHBoxLayout()
		for widget in [self.label_zoomxy, self.checkbox_zoomxy]:
			layout_zoomxy.addWidget(widget)
		layout_width = QHBoxLayout()
		for widget in [self.label_width, self.slider_width, self.edit_width]:
			layout_width.addWidget(widget)
		layout_height = QHBoxLayout()
		for widget in [self.label_height, self.slider_height, self.height_edit]:
			layout_height.addWidget(widget)
		layout_offsetx = QHBoxLayout()
		for widget in [self.label_offsetx, self.slider_offsetx, self.edit_offsetx]:
			layout_offsetx.addWidget(widget)
		layout_offsety = QHBoxLayout()
		for widget in [self.label_offsety, self.slider_offsety, self.edit_offsety]:
			layout_offsety.addWidget(widget)

		layout_main = QVBoxLayout()
		for layout in [layout_zoomxy, layout_width, layout_height, layout_offsetx, layout_offsety]:
			layout.setAlignment(Qt.AlignLeft)
			layout_main.addLayout(layout)
		layout_main.addWidget(self.button)

		self.setLayout(layout_main)

		self.setWindowTitle("Adjust Camera")
		self.setMinimumWidth(250)
		self.adjustSize()
		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height * 3 // 4)

	def set_width(self, width_new, original_call=True):
		width_new = width_new // 4 * 4

		self.slider_width.setValue(width_new)
		self.edit_width.setText(str(width_new))

		offsetx_max_new = self.max_width_orig - width_new
		self.slider_offsetx.setMaximum(offsetx_max_new)
		self.edit_offsetx.setValidator(MultiplesValidator(4, 0, offsetx_max_new))

		offsetx_new = min(offsetx_max_new, int(self.edit_offsetx.text()))
		self.slider_offsetx.setValue(offsetx_new)
		self.edit_offsetx.setText(str(offsetx_new))

		self.camera.StopGrabbing()
		self.camera.Width.SetValue(width_new)
		self.camera.OffsetX.SetValue(offsetx_new)
		self.camera.StartGrabbing(GrabStrategy_LatestImageOnly)

		if original_call and self.checkbox_zoomxy.isChecked():
			self.set_height(width_new, False)

	def set_height(self, height_new, original_call=True):
		self.slider_height.setValue(height_new)
		self.height_edit.setText(str(height_new))

		offsety_max_new = (self.max_dim - height_new) // 2 * 2
		self.slider_offsety.setMaximum(offsety_max_new)
		self.edit_offsety.setValidator(MultiplesValidator(2, 0, offsety_max_new))

		offsety_new = min(offsety_max_new, int(self.edit_offsety.text()))
		self.slider_offsety.setValue(offsety_new)
		self.edit_offsety.setText(str(offsety_new))

		self.camera.StopGrabbing()
		self.camera.Height.SetValue(height_new)
		self.camera.OffsetY.SetValue(offsety_new)
		self.camera.StartGrabbing(GrabStrategy_LatestImageOnly)

		if original_call and self.checkbox_zoomxy.isChecked():
			self.set_width(height_new, False)

	def set_offsetx(self, offsetx_new):
		offsetx_new = offsetx_new // 4 * 4

		self.slider_offsetx.setValue(offsetx_new)
		self.edit_offsetx.setText(str(offsetx_new))

		self.camera.StopGrabbing()
		self.camera.OffsetX.SetValue(offsetx_new)
		self.camera.StartGrabbing(GrabStrategy_LatestImageOnly)

	def set_offsety(self, offsety_new):
		offsety_new = offsety_new // 2 * 2

		self.slider_offsety.setValue(offsety_new)
		self.edit_offsety.setText(str(offsety_new))

		self.camera.StopGrabbing()
		self.camera.OffsetY.SetValue(offsety_new)
		self.camera.StartGrabbing(GrabStrategy_LatestImageOnly)

	def closeEvent(self, event):
		QApplication.exit(0)


def main():
	# camera setup, connects to the first Basler camera found. If it falils, an error popup and exits
	try:
		camera = InstantCamera(TlFactory.GetInstance().CreateFirstDevice())
	except:
		show_error('Cannot access the camera. Make sure all other software that accesses it is closed.')
		return -1

	# open the camera, veryfies it and applies the project setting defined in utils.py
	camera.Open()
	if not camera.IsOpen():
		show_error('Cannot find camera.')
		return -1
	setup_camera(camera) # set upt the camera based on the utils.py

	# window setup
	monitors = get_monitors()

	# Creates a square control window on the primary monitor, sized to 80% of the screen height, offset 10% from the corner.
	controlwindow_name = 'Control Window'
	namedWindow(controlwindow_name, WINDOW_NORMAL)
	resizeWindow(controlwindow_name, monitors[0].height * 8 // 10, monitors[0].height * 8 // 10)
	moveWindow(controlwindow_name, monitors[0].x + monitors[0].height * 1 // 10, monitors[0].y + monitors[0].height * 1 // 10)

	# If a second monitor (your projector) exists: creates a "Projection Window", moves it onto the projector, and makes it fullscreen. Otherwise, a popup warns you.
	if len(monitors) > 1:
		projwindow_name = 'Projection Window'
		namedWindow(projwindow_name, WINDOW_NORMAL)
		moveWindow(projwindow_name, monitors[1].x, monitors[1].y)
		resizeWindow(projwindow_name, monitors[1].width, monitors[1].height)
		setWindowProperty(projwindow_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)
	else:
		show_info('No second screen detected!')

	# parameter dialog setup, starts the Qt application and shows the Adjust Camera dialog.
	app = QApplication(argv)
	controller = ParameterControl(camera)
	controller.show()

	# main loop, run this while: the camera is grabbing, the Qt dialog is open, and the OpenCV control window hasn't been closed by the user.
	camera.StartGrabbing(GrabStrategy_LatestImageOnly)
	while camera.IsGrabbing() and controller.isVisible() and getWindowProperty(controlwindow_name, WND_PROP_VISIBLE):

		# Waits up to 1000 ms for a frame; throws an exception on timeout; breaks the loop on grab failure.
		grabResult = camera.RetrieveResult(1000, TimeoutHandling_ThrowException)
		if not grabResult.GrabSucceeded(): break

		# Converts the grab result to a NumPy array, flips it both horizontally and vertically
		# (-1 = 180° rotation — presumably to match the physical camera/projector orientation), and releases the buffer back to the driver.
		frame = flip(grabResult.Array, -1)
		grabResult.Release()

		#If the frame is color (3 channels), convert to grayscale.
		if len(frame.shape) != 2:
			frame = cvtColor(frame, COLOR_BGR2GRAY)

		# Stretches the frame into a square (side = larger dimension). Note: this distorts non-square ROIs.
		image_square_size = max(frame.shape)
		image = resize(frame, (image_square_size, image_square_size))
		# Shows the square image in the control window
		imshow(controlwindow_name, image)

		# For the projector: computes how much black padding to add left and right so the square image,
		# when stretched to the projector's (typically 16:9) fullscreen window, keeps its square proportions. Pads with zeros (black) and displays it.
		if len(monitors) > 1:
			pad_width = round((image_square_size * monitors[1].width / monitors[1].height - image_square_size) / 2)
			image = pad(image, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
			imshow(projwindow_name, image)
		waitKey(1)

	# Cleanup, stops acquisition, releases the camera, closes all OpenCV windows and the Qt app.
	camera.StopGrabbing()
	camera.Close()
	destroyAllWindows()
	controller.close()
	app.quit()

	return 0


if __name__ == '__main__':
	print('PROGRAM STARTED')
	main()
