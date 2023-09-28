from pycgp.ip_utils import *
import numpy as np
import math
import cv2
from scipy.stats import kurtosis, skew
from skimage.morphology import remove_small_objects, remove_small_holes, thin
import skimage.feature as feature
from pycgp import CGP
import mahotas

def build_funcLib():
    return [
 			CGP.CGPFunc(f_erode,'erode',1,2),
            CGP.CGPFunc(f_dilate,'dilate',1,2),
            CGP.CGPFunc(f_open,'f_open',1,2),
            CGP.CGPFunc(f_close,'f_close',1,2),
            CGP.CGPFunc(f_morph_gradient,'f_morph_gradient',1,2),
            CGP.CGPFunc(f_morph_top_hat,'f_morph_top_hat',1,2),
            CGP.CGPFunc(f_morph_black_hat,'f_morph_black_hat',1,2),
            CGP.CGPFunc(f_fill_holes,'f_fill_holes',1,0),
            CGP.CGPFunc(f_remove_small_holes,'f_remove_small_holes',1,1),
            CGP.CGPFunc(f_remove_small_objects,'f_remove_small_objects',1,1),
            CGP.CGPFunc(f_median_blur,'f_median_blur',1,1),
            CGP.CGPFunc(f_gaussian_blur,'f_gaussian_blur',1,1),
            CGP.CGPFunc(f_laplacian,'f_laplacian',1,0),
            CGP.CGPFunc(f_sobel,'f_sobel',1,2),
            CGP.CGPFunc(f_robert_cross,'f_robert_cross',1,1),
            CGP.CGPFunc(f_canny,'f_canny',1,2),
            CGP.CGPFunc(f_sharpen,'f_sharpen',1,0),
            CGP.CGPFunc(f_absolute_difference,'f_absolute_difference',1,2),
            CGP.CGPFunc(f_absolute_difference,'f_absolute_difference2',2,0),
            CGP.CGPFunc(f_relative_difference,'f_relative_difference',1,1),
            CGP.CGPFunc(f_fluo_top_hat,'f_fluo_top_hat',1,2),
            CGP.CGPFunc(f_gabor_filter,'f_gabor_filter',1,2),
            CGP.CGPFunc(f_distance_transform,'f_distance_transform',1,1),
            CGP.CGPFunc(f_distance_transform_and_thresh,'f_distance_transform_and_thresh',1,2),
            CGP.CGPFunc(f_threshold,'f_threshold',1,2),
            CGP.CGPFunc(f_threshold_at_1,'f_threshold_at_1',1,1),

            CGP.CGPFunc(f_binary_in_range,'f_binary_in_range',1,2),
            CGP.CGPFunc(f_in_range,'f_in_range',1,2),
            CGP.CGPFunc(f_bitwise_and,'f_bitwise_and',2,0),
            CGP.CGPFunc(f_bitwise_and_mask,'f_bitwise_and_mask',2,0),
            CGP.CGPFunc(f_bitwise_not,'f_bitwise_not',1,0),
            CGP.CGPFunc(f_bitwise_or,'f_bitwise_or',2,0),
            CGP.CGPFunc(f_bitwise_xor,'f_bitwise_xor',2,0),

            CGP.CGPFunc(f_add,'f_add',2,0),
            CGP.CGPFunc(f_sub,'f_subtract',2,0),

            CGP.CGPFunc(f_square_root,'f_square_root',1,0),
            CGP.CGPFunc(f_square,'f_square',1,0),
            CGP.CGPFunc(f_exp,'f_exp',1,0),
            CGP.CGPFunc(f_log,'f_log',1,0),
            CGP.CGPFunc(f_min, 'f_min', 2, 0),
            CGP.CGPFunc(f_max, 'f_max', 2, 0),
            CGP.CGPFunc(f_mean, 'f_mean', 2, 0),

            CGP.CGPFunc(f_kirsch, 'f_kirsch', 1, 0), #OK
            CGP.CGPFunc(f_embossing, 'f_embossing', 1, 0), #OK
            CGP.CGPFunc(f_pyr, 'f_pyr', 1, 1), #OK
            CGP.CGPFunc(f_denoizing, 'f_denoizing', 1, 1), #OK
            CGP.CGPFunc(f_threshold_otsu, 'f_threshold_otsu', 1, 0), #OK

			CGP.CGPFunc(f_contour_area, 'f_contour_area', 1, 1),
	]

def _fill_cnt(image, contours):
	assert (
			len(image.shape) == 3 or len(image.shape) == 2
	), "given image wrong format, shape must be (h, w, c) or (h, w)"
	if len(image.shape) == 3 and image.shape[-1] == 3:
		color = IMAGE_UINT8_COLOR_3C
	elif len(image.shape) == 2:
		color = IMAGE_UINT8_COLOR_1C
	else:
		raise ValueError("Image wrong format, must have 1 or 3 channels")
	selected = -1  # selects all the contours (-1)
	thickness = -1  # fills the contours (-1)
	final_img = cv2.drawContours(image, contours, selected, color, thickness)
	#print(f'{final_img.max()}, {final_img.sum()}, {final_img.mean()}')
	return final_img

def f_contour_area(args, const_params):
	# Contours find
	threshold_scaler = 9 # threshold between 0 and 40*255=10200

	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for c in contours:
		if const_params[0] > 128:
			if cv2.contourArea(c) >= const_params[0]-128:
				final_contours.append(c)
		else:
			if cv2.contourArea(c) <= 128 - const_params[0]:
				final_contours.append(c)

	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_contrast(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		masked_img = _fill_cnt(image.copy(), [cnt]) * image
		sub_img = masked_img[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i*np.pi/8 for i in range(8)], levels=256)
		contrast = feature.graycoprops(graycom, 'contrast').mean()
		threshold = (128-const_params[1]) / 128
		if threshold > 0:
			if contrast > threshold:
				final_contours.append(cnt)
		else:
			if contrast < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_dissimilarity(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i*np.pi/8 for i in range(8)], levels=256)
		dissimilarity = feature.graycoprops(graycom, 'dissimilarity').mean()
		threshold = (128-const_params[1]) / 128
		if threshold > 0:
			if dissimilarity > threshold:
				final_contours.append(cnt)
		else:
			if dissimilarity < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_homogeneity(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		homogeneity = feature.graycoprops(graycom, 'homogeneity').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if homogeneity > threshold:
				final_contours.append(cnt)
		else:
			if homogeneity < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_ASM(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		asm = feature.graycoprops(graycom, 'ASM').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if asm > threshold:
				final_contours.append(cnt)
		else:
			if asm < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_energy(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		energy = feature.graycoprops(graycom, 'energy').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if energy > threshold:
				final_contours.append(cnt)
		else:
			if energy < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]


def f_contour_correlation(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		correlation = feature.graycoprops(graycom, 'correlation').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if correlation > threshold:
				final_contours.append(cnt)
		else:
			if correlation < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]


def f_local_binary_pattern(args, const_params):
	point_scaler = 0.2 # max point 51
	radius_scaler = 0.04 # max radius 10
	return feature.local_binary_pattern(args[0].copy(), const_params[0], const_params[1])

def f_add(args, const_params):
	return cv2.add(args[0], args[1])

def f_sub(args, const_params):
	return cv2.subtract(args[0], args[1])

def f_mean(args, const_params):
    return cv2.addWeighted(args[0], 0.5, args[1], 0.5, 0)

def f_bitwise_not(args, const_params):
	return cv2.bitwise_not(args[0])

def f_bitwise_or(args, const_params):
	return cv2.bitwise_or(args[0], args[1])

def f_bitwise_and(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	y = args[1].copy()
	y = y.astype(np.uint8)
	return cv2.bitwise_and(x, y)

def f_bitwise_and_mask(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	y = args[1].copy()
	y = y.astype(np.uint8)
	return cv2.bitwise_and(x, x, mask=y)

def f_bitwise_xor(args, const_params):
	return cv2.bitwise_xor(args[0], args[1])


def f_square_root(args, const_params):
	return (cv2.sqrt((args[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)


def f_square(args, const_params):
	return (cv2.pow((args[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


def f_exp(args, const_params):
	return (cv2.exp((args[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)

def f_log(args, const_params):
	return np.log1p(args[0]).astype(np.uint8)


def f_median_blur(args, const_params):
	ksize = correct_ksize(const_params[0])
	x = args[0].copy()
	x = x.astype(np.uint8)
	return cv2.medianBlur(x, ksize)

def f_gaussian_blur(args, const_params):
	ksize = correct_ksize(const_params[0])
	return cv2.GaussianBlur(args[0], (ksize, ksize), 0)

def f_laplacian(args, const_params):
	return cv2.Laplacian(args[0], cv2.CV_64F).astype(np.uint8)


def f_sobel(args, const_params):
	ksize = correct_ksize(const_params[0])
	if const_params[1] < 128:
		return cv2.Sobel(args[0], cv2.CV_64F, 1, 0, ksize=ksize).astype(np.uint8)
	return cv2.Sobel(args[0], cv2.CV_64F, 0, 1, ksize=ksize).astype(np.uint8)


def f_robert_cross(args, const_params):
	img = (args[0] / 255.0).astype(np.float32)
	h = cv2.filter2D(img, -1, ROBERT_CROSS_H_KERNEL)
	v = cv2.filter2D(img, -1, ROBERT_CROSS_V_KERNEL)
	return (cv2.sqrt(cv2.pow(h, 2) + cv2.pow(v, 2)) * 255).astype(np.uint8)

def f_canny(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	return cv2.Canny(x, const_params[0], const_params[1])

def f_sharpen(args, const_params):
	return cv2.filter2D(args[0], -1, SHARPEN_KERNEL)

def f_gabor_filter(args, const_params):
    ksize = 11
    gabor_k = gabor_kernel(ksize, const_params[0], const_params[1])
    return cv2.filter2D(args[0], -1, gabor_k)


def f_absolute_difference(args, const_params):
	ksize = correct_ksize(const_params[0])
	image = args[0].copy()
	return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + const_params[1]

def f_absolute_difference2(args, const_params):
    return 255 - cv2.absdiff(args[0], args[1])

def f_fluo_top_hat(args, const_params):
	kernel = kernel_from_parameters(const_params)
	img = cv2.morphologyEx(args[0], cv2.MORPH_TOPHAT, kernel, iterations=10)
	kur = np.mean(kurtosis(img, fisher=True))
	skew1 = np.mean(skew(img))
	if kur > 1 and skew1 > 1:
		p2, p98 = np.percentile(img, (15, 99.5), interpolation="linear")
	else:
		p2, p98 = np.percentile(img, (15, 100), interpolation="linear")
	# rescale intensity
	output_img = np.clip(img, p2, p98)
	if p98 - p2 == 0:
		return (output_img * 255).astype(np.uint8)
	output_img = (output_img - p2) / (p98 - p2) * 255
	return output_img.astype(np.uint8)

def f_relative_difference(args, const_params):
	img = args[0]
	max_img = np.max(img)
	min_img = np.min(img)

	ksize = correct_ksize(const_params[0])
	gb = cv2.GaussianBlur(img, (ksize, ksize), 0)
	gb = np.float32(gb)

	img = np.divide(img, gb + 1e-15, dtype=np.float32)
	img = cv2.normalize(img, img, max_img, min_img, cv2.NORM_MINMAX)
	return img.astype(np.uint8)


def f_erode(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.erode(args[0], kernel)

def f_dilate(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.dilate(args[0], kernel)

def f_open(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_OPEN, kernel)

def f_close(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_CLOSE, kernel)

def f_morph_gradient(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_GRADIENT, kernel)

def f_morph_top_hat(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_TOPHAT, kernel)

def f_morph_black_hat(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_BLACKHAT, kernel)

def f_fill_holes(args, const_params):
    # Contours find
    image = args[0]
    method = cv2.CHAIN_APPROX_SIMPLE
    mode = cv2.RETR_EXTERNAL
    contours = cv2.findContours(image.copy(), mode, method)[0]
    
    assert (
        len(image.shape) == 3 or len(image.shape) == 2
    ), "given image wrong format, shape must be (h, w, c) or (h, w)"
    if len(image.shape) == 3 and image.shape[-1] == 3:
        color = IMAGE_UINT8_COLOR_3C
    elif len(image.shape) == 2:
        color = IMAGE_UINT8_COLOR_1C
    else:
        raise ValueError("Image wrong format, must have 1 or 3 channels")
    selected = -1  # selects all the contours (-1)
    thickness = -1  # fills the contours (-1)
    return cv2.drawContours(image.copy(), contours, selected, color, thickness)

def f_remove_small_objects(args, const_params):
	return remove_small_objects(args[0] > 0, const_params[0]).astype(np.uint8)


def f_remove_small_holes(args, const_params):
	return remove_small_holes(args[0] > 0, const_params[0]).astype(np.uint8)

def f_threshold(args, const_params):
        if const_params[0] < 128:
            return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE,  cv2.THRESH_BINARY)[1]
        return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE, cv2.THRESH_TOZERO)[1]

def f_threshold_at_1(args, const_params):
    if const_params[0] < 128:
        return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE,  cv2.THRESH_BINARY)[1]
    return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE, cv2.THRESH_TOZERO)[1]


def f_distance_transform(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	return cv2.normalize(
		cv2.distanceTransform(x, cv2.DIST_L2, 3),
		None,
		0,
		255,
		cv2.NORM_MINMAX,
		cv2.CV_8U,
	)

def f_distance_transform_and_thresh(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)

	d = cv2.normalize(
		cv2.distanceTransform(x, cv2.DIST_L2, 3),
		None,
		0,
		255,
		cv2.NORM_MINMAX,
		cv2.CV_8U,
	)
	return cv2.threshold(d, const_params[1], IMAGE_UINT8_POSITIVE,  cv2.THRESH_BINARY)[1]


def f_binary_in_range(args, const_params):
	lower = int(min(const_params[0], const_params[1]))
	upper = int(max(const_params[0], const_params[1]))
	return cv2.inRange(args[0], lower, upper)


def f_in_range(args, const_params):
	lower = int(min(const_params[0], const_params[1]))
	upper = int(max(const_params[0], const_params[1]))
	return cv2.bitwise_and(
		args[0],
		args[0],
		mask=cv2.inRange(args[0], lower, upper),
        )


def f_min(args, const_params):
    return cv2.min(args[0], args[1])


def f_max(args, const_params):
    return cv2.max(args[0], args[1])

def f_pyr(args, const_params):
    if const_params[0] < 128:
        h, w = args[0].shape
        scaled_half = cv2.pyrDown(args[0])
        return cv2.resize(scaled_half, (w, h))
    else:
        h, w = args[0].shape
        scaled_twice = cv2.pyrUp(args[0])
        return cv2.resize(scaled_twice, (w, h))

def f_kirsch(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    compass_gradients = [
        cv2.filter2D(x, ddepth=cv2.CV_32F, kernel=kernel/5)
        for kernel in KERNEL_KIRSCH_COMPASS
    ]
    res = np.max(compass_gradients, axis=0)
    res[res > 255] = 255
    return res.astype(np.uint8)

def f_embossing(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    res = cv2.filter2D(x, ddepth=cv2.CV_32F, kernel=KERNEL_EMBOSS)
    res[res > 255] = 255
    return res.astype(np.uint8)

def f_normalization(args, const_params):
    return cv2.normalize(args[0],  None, 0, 255, cv2.NORM_MINMAX)

def f_denoizing(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    return cv2.fastNlMeansDenoising(x, None, h=np.uint8(const_params[0]))

def f_threshold_otsu(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    return cv2.threshold(x, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]
