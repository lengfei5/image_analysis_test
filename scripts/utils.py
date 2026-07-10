from os import getcwd, listdir, makedirs
from os.path import join, split
from numpy import array, mean, sin, cos, sqrt, arctan2, uint8, float16
from tkinter import Tk
from tkinter.messagebox import showinfo, showerror
from tkinter.filedialog import asksaveasfilename, askopenfilename, askdirectory
from cv2 import waitKey, imshow, imread, imwrite, cvtColor, COLOR_BGR2GRAY


# conversion
def pol2cart(r, phi):
    x = r * cos(phi)
    y = r * sin(phi)
    return (x, y)

def cart2pol(x, y):
	r = sqrt(x**2 + y**2)
	phi = arctan2(y, x)
	return (r, phi)


# opencv
def show_image(image, window='OpenCV', duration=0):
    image = imshow('image' if not window else window, image)
    waitKey(duration)


# gui
def show_info(message=None):
	root = Tk()
	root.withdraw()
	showinfo('Information', message)

def show_error(message=None):
	root = Tk()
	root.withdraw()
	showerror('Error', message)

def save_file(title=None, initialdir='.', filetypes=['*'], defaultextension='txt', initialfile='file.txt'):
	root = Tk()
	root.withdraw()
	file_path = asksaveasfilename(title=title, initialdir=initialdir, defaultextension=defaultextension, filetypes=filetypes, initialfile=initialfile)
	return file_path

def choose_file(title=None):
	root = Tk()
	root.withdraw()
	file_path = askopenfilename(title=title)
	return file_path

def choose_directory(title=None):
	root = Tk()
	root.withdraw()
	dir_path = askdirectory(title=title)
	return dir_path


# other
def setup_camera(camera):
	if camera.Width.GetValue() > camera.HeightMax.Value:
		camera.Width.SetValue(camera.HeightMax.Value // 4 * 4)

	camera.PixelFormat.Value = "Mono8"

	camera.AcquisitionFrameRateEnable.SetValue(True)
	camera.AcquisitionFrameRate.SetValue(30)

	camera.ExposureTime.Value = 20000

	camera.BinningHorizontalMode.SetValue('Sum')
	camera.BinningVerticalMode.SetValue('Sum')
	camera.BinningHorizontal.SetValue(1)
	camera.BinningVertical.SetValue(1)

	camera.CenterX.SetValue(False)
	camera.CenterY.SetValue(False)


def extract_background():
	fps = 30
	frequency = 15
	start = 0 #- 7000
	stop = -1 #- 7000

	print('\n')
	path = choose_directory()

	print('Reading...')
	img_names = sorted(listdir(path), key=lambda name: int(name.split('_')[-1].split('.')[0]))

	print('Importing...')
	imgs = [cvtColor(imread(join(path, img_name)), COLOR_BGR2GRAY) for img_name in img_names[start:stop:(fps // frequency)]]

	print('Computing...')
	mean_img = mean(array(imgs).astype(float16), axis=0).astype(uint8)

	print('Saving...')
	makedirs(join(getcwd(), 'data'), exist_ok=True)
	suffix = f'_{split('/')[-1]}'
	imwrite(join(getcwd(), 'data', f'background{suffix}.jpg'), mean_img)

	print('Done!')
