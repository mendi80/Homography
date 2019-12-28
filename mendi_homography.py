import cv2
import numpy as np


def imshow_scaled_window(winname, img, display_width):
	H, W, CH = img.shape
	scale = display_width / W
	display_height = H * scale
	cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
	cv2.imshow(winname, img)
	cv2.resizeWindow(winname, (round(display_width), round(display_height)))
	# cv2.moveWindow(winname, 100, 30)
	return scale


def mous_callback(event, x, y, flags, data):
	if event == cv2.EVENT_LBUTTONDOWN:
		if len(data['points']) < 4:
			# cv2.circle(image, center_coordinates, radius, color, thickness)
			cv2.circle(data['img'], (x, y), round(6 / data['scale']), (0, 0, 255), round(6 / data['scale']), 16)
			cv2.imshow(data['winname'], data['img'])
			data['points'].append([x, y])
		else:
			print("done! press Enter")


def get_4points(winname, img, display_width):
	scale = imshow_scaled_window(winname, img, display_width)
	data = {'winname': winname, 'img': img.copy(), 'scale': scale, 'points': []}
	cv2.setMouseCallback(winname, mous_callback, data)
	print(f"Click on 4 points on '{winname}' window.", "Order= Top-Left, Top-Right, Bottom-Left, Bottom-Right")
	cv2.waitKey(0)
	print(f"Done. Points collected={data['points']}")
	return np.vstack(data['points']).astype(float)


def rectify(source_file, target_width, target_height, display_width=800):
	source_image = cv2.imread(source_file)
	source_points = get_4points("Source", source_image, display_width)  # CW: LT, RT, BR, BL
	target_points = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=float)
	H, status = cv2.findHomography(source_points, target_points)
	target_image = cv2.warpPerspective(source_image, H, (target_width, target_height))  # np.zeros(target_shape, np.uint8)
	imshow_scaled_window("Result", target_image, display_width)


# cv2.moveWindow("Result", 400, 300)

def paste(source_file, target_file, display_width=800):
	source_image = cv2.imread(source_file)
	imshow_scaled_window("Source", source_image, display_width / 2)
	source_height, source_width, _ = source_image.shape
	source_points = np.array([[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]], dtype=float)

	target_image = cv2.imread(target_file)
	target_height, target_width, _ = source_image.shape
	target_points = get_4points("Target", target_image, display_width)

	H, status = cv2.findHomography(source_points, target_points)
	patch_image = cv2.warpPerspective(source_image, H, (target_width, target_height))

	cv2.fillConvexPoly(target_image, target_points.astype(int), 0, 16)
	target_image += patch_image
	imshow_scaled_window("Target", target_image, display_width)


def copypaste(source_file, target_file, display_width=800):
	source_image = cv2.imread(source_file)
	imshow_scaled_window("Source", source_image, display_width / 2)
	source_height, source_width, _ = source_image.shape
	source_points = get_4points("Source", source_image, display_width)

	source_points_width = max([point[0] for point in source_points]) - min([point[0] for point in source_points])
	source_points_height = max([point[1] for point in source_points]) - min([point[1] for point in source_points])
	patch1_width, patch1_height = int(source_points_width * 4), int(source_points_height * 4)
	patch1_points = np.array([[0, 0], [patch1_width - 1, 0], [patch1_width - 1, patch1_height - 1], [0, patch1_height - 1]], dtype=float)

	H1, status1 = cv2.findHomography(source_points, patch1_points)
	patch1_image = cv2.warpPerspective(source_image, H1, (patch1_width, patch1_height))

	target_image = cv2.imread(target_file)
	target_height, target_width, _ = source_image.shape
	target_points = get_4points("Target", target_image, display_width)

	H2, status2 = cv2.findHomography(patch1_points, target_points)
	patch2_image = cv2.warpPerspective(patch1_image, H2, (target_width, target_height))

	cv2.fillConvexPoly(target_image, target_points.astype(int), 0, 16)
	target_image += patch2_image
	imshow_scaled_window("Target", target_image, display_width)


file1 = r"image1.jpg"
file2 = r"image2.jpg"

PROCESS = 'rectify'
PROCESS = 'paste'
PROCESS = 'copypaste'

if PROCESS == 'rectify': rectify(file2, 512, 512, 1000)
if PROCESS == 'paste': paste(file1, file2, 1000)
if PROCESS == 'copypaste': copypaste(file1, file2, 1000)
