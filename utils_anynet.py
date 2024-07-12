#! /usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import shutil
import time
import math

# TO-DO=>
# - jsonize config files
# - fix git tracking issue


import matplotlib.pyplot as plt




# input -> np.float32 disp_data
def get_depth_data(disp_data, baseline, focal_length): 
	assert disp_data.dtype == np.float32
	depth_data = (baseline * focal_length) / (disp_data + 1e-6)
	return depth_data 	


def reject_outliers_2(data, m=2.):
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d / (mdev if mdev else 1.)
	mask = s < m
	return np.where(mask, data, np.nan)
	# return data[s < m]

def normalization_percentile(arr, lo= 2., hi=98.):
	arr_min, arr_max = np.percentile(arr, (2, 98))  # Use 2nd and 98th percentiles
	norm_perc_arr = (arr - arr_min) / (arr_max - arr_min)
	return norm_perc_arr


def normalization_log(arr):
	if np.min(arr) < 0:
		arr = arr - np.min(arr)

	log_arr= np.log1p(arr)
	norm_log_arr = (log_arr - log_arr.min()) / (log_arr.max() - log_arr.min())
	# logging.info(f"norm_log_arr.shape: {norm_log_arr.shape}")
	return norm_log_arr

# removes inf values and normalizes to 255
def uint8_normalization(depth_map):
	assert depth_map.dtype == np.float32
	max_depth = np.max(depth_map[np.isfinite(depth_map)])
	depth_map_finite = np.where(np.isinf(depth_map), max_depth, depth_map)
	depth_map_uint8 = cv2.normalize(depth_map_finite, depth_map_finite, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return depth_map_uint8



def write_legend_plot(depth_error_data, save_location):
	fig, axs = plt.subplots(1, 2, figsize=(20, 10))
	cax1 = axs[0].imshow(depth_error_data, cmap='inferno')
	cbar1 = fig.colorbar(cax1, ax=axs[0])
	cbar1.set_label('Depth Error (Inferno)')
	cax2 = axs[1].imshow(depth_error_data, cmap='gray')
	cbar2 = fig.colorbar(cax2, ax=axs[1])
	cbar2.set_label('Depth Error (Grayscale)')
	plt.savefig(save_location)
	plt.close(fig)


def inf_filtering(depth_map):
	max_depth = np.max(depth_map[np.isfinite(depth_map)])
	depth_map_finite = np.where(np.isinf(depth_map), max_depth, depth_map)
	return depth_map_finite

# crops the upper height percent of the image
def crop_image(image, height_percent, width_percent):
	height, width = image.shape[:2]
	new_width = int(width * width_percent)
	new_height = int(height * height_percent)
	logging.debug(f"(new_height, new_width): ({new_height}, {new_width}")
	cropped_image = image[new_height:height, 0:new_width]
	logging.debug(f"cropped_image.shape: {cropped_image.shape}")
	return cropped_image

def is_grayscale(image):
	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	grayscale_image_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
	return np.array_equal(image, grayscale_image_bgr)

def percentage_infinite_points(image):
	total_points = image.size
	infinite_points = np.sum(np.isinf(image))
	percentage = (infinite_points / total_points) * 100
	return percentage

# def save_npy_as_ply(ply_file: str, color : np.ndarray, array : np.ndarray) -> None:
# 	'''
# 	Save a numpy array as a PLY file.
# 	args:
# 		ply_file: str: path to the PLY file
# 		array: np.ndarray: the array to save
# 	'''
# 	with open(ply_file, 'w') as f:
# 		f.write("ply\n")
# 		f.write("format ascii 1.0\n")
# 		f.write(f"element vertex {len(array)}\n")
# 		f.write("property float x\n")
# 		f.write("property float y\n")
# 		f.write("property float z\n")
# 		f.write("end_header\n")
# 		for point in array:
# 			f.write(f"{point[0]} {point[1]} {point[2]}\n")

def save_npy_as_ply(ply_file: str, array: np.ndarray, colors: np.ndarray) -> None:
	# logging.warning(f"colors.shape: {colors.shape} colors.dtype: {colors.dtype}")
	# logging.warning(f"colors[:30]: {colors[:30]}")
	
	'''
	Save a numpy array as a PLY file with color information.
	args:
		ply_file: str: path to the PLY file
		array: np.ndarray: the array to save, expected to have shape (n, 3) with each row being [x, y, z]
		colors: np.ndarray: the colors array, expected to have shape (n, 3) with each row being [r, g, b]
	'''
	if len(array) != len(colors):
		raise ValueError("Array and colors must have the same length")

	with open(ply_file, 'w') as f:
		f.write("ply\n")
		f.write("format ascii 1.0\n")
		f.write(f"element vertex {len(array)}\n")
		f.write("property float x\n")
		f.write("property float y\n")
		f.write("property float z\n")
		# Add properties for color
		f.write("property uchar red\n")
		f.write("property uchar green\n")
		f.write("property uchar blue\n")
		f.write("end_header\n")
		for point, color in zip(array, colors):
			# Ensure color values are integers in the range [0, 255]
			color = color.astype(int)
			f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")


def delete_folders(folders):
	for folder_path in folders:
			logging.debug(f"Deleting the old files in {folder_path}")
			if os.path.exists(folder_path):
				try: 
					shutil.rmtree(folder_path)
				except OSError:
					logging.error(f"Error while deleting {folder_path}. Retrying...")
					# time.sleep(1)  # wait for 1 second before retrying
			else:
				print(f"The folder {folder_path} does not exist.")	
	logging.warning(f"Deleted the old files in {folders}!")
		
def create_folders(folders):
	for path in folders:
		os.makedirs(path, exist_ok=True)
		logging.warning(f"Created the {path} folder!")
