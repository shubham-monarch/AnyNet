#! /usr/bin/env python3

'''script to prepare the original sceneflow dataset for finetune.py'''

import os
import shutil
from glob import glob
import logging, coloredlogs
from tqdm import tqdm
import cv2
import numpy as np


# custom imports
from dataloader import readpfm
import utils_anynet

# Define sceneflow dataset folders
DATASET_FOLDER = "dataset-anynet"
LEFT_IMAGES_FOLDER = f"{DATASET_FOLDER}/image_2"
RIGHT_IMAGES_FOLDER = f"{DATASET_FOLDER}/image_3"
LEFT_DISPARITY_FOLDER = f"{DATASET_FOLDER}/disp_occ_0"

# Define train and validation folders
DATASET_FOLDER_FINETUNE = "dset-finetune"
TRAIN_FOLDER = f"{DATASET_FOLDER_FINETUNE}/training"
VALIDATION_FOLDER = f"{DATASET_FOLDER_FINETUNE}/validation"

TRAIN_LEFT_FOLDER = f"{TRAIN_FOLDER}/image_2/"
TRAIN_RIGHT_FOLDER = f"{TRAIN_FOLDER}/image_3/"
TRAIN_DISPARITY_FOLDER = f"{TRAIN_FOLDER}/disp_occ_0/"

VALIDATION_LEFT_FOLDER = f"{VALIDATION_FOLDER}/image_2/"
VALIDATION_RIGHT_FOLDER = f"{VALIDATION_FOLDER}/image_3/"
VALIDATION_DISPARITY_FOLDER = f"{VALIDATION_FOLDER}/disp_occ_0/"


# Folders to create
FOLDERS_TO_CREATE = [DATASET_FOLDER_FINETUNE]

# split dset into test / validation folders
def split_to_train_validation(source_folder, dest_train_folder, dest_validation_folder, split_index):
	files = sorted(glob(f"{source_folder}/*"))
	train_files = files[:split_index]
	validation_files = files[split_index:]

	os.makedirs(dest_train_folder, exist_ok=True)
	os.makedirs(dest_validation_folder, exist_ok=True)

	for file in tqdm(train_files, desc="Copying train files"):
		shutil.copy(file, dest_train_folder)
		
	for file in tqdm(validation_files, desc="Copying validation files"):
		shutil.copy(file, dest_validation_folder)

# generating png-disparity from pfm-disparity 
def process_disp_folder(source_folder, dest_folder = None):
	files = sorted(glob(f"{source_folder}/*"))
	logging.warning(f"Number of files in {os.path.dirname(source_folder)}: {len(files)}")
	
	if dest_folder is None:
		dest_folder = f"{os.path.dirname(source_folder)}-png"
		utils_anynet.create_folders([dest_folder])
	
	logging.warning(f"Converting .pfm files to .png files in {[dest_folder]}")
	
	for file in tqdm(files):
		data, _ = readpfm.readPFM(file)
		data *= 256

		data_normalized = cv2.normalize(data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		data_uint8 = np.uint8(data_normalized)

		filename = os.path.basename(file)
		cv2.imwrite(f"{dest_folder}/{filename.replace('.pfm', '.png')}", data_uint8)
	
# preparing png-disparity files for finetune.py
def process_dset(source_folder, target_file=None):
	
	if target_file is None:
		# renaming the disparity folders to load png disparities
		disp_pfm_old = os.path.join(source_folder, "disp_occ_0/")
		disp_pfm_new = os.path.join(source_folder, "disp_occ_0-pfm/")
		utils_anynet.create_folders([disp_pfm_new])
		disp_png = os.path.join(source_folder, "disp_occ_0-png/")
		
		os.rename(disp_pfm_old, disp_pfm_new)
		os.rename(disp_png, disp_pfm_old)
		# os.rename(temp_path_pfm, disp_png)

def main():
	
	utils_anynet.delete_folders(FOLDERS_TO_CREATE)
	utils_anynet.create_folders(FOLDERS_TO_CREATE)

	left_images_count = len(os.listdir(LEFT_IMAGES_FOLDER))
	right_images_count = len(os.listdir(RIGHT_IMAGES_FOLDER))
	left_disparity_count = len(os.listdir(LEFT_DISPARITY_FOLDER))

	assert left_images_count == right_images_count == left_disparity_count, "The number of files in the folders are not the same."

	# Calculate split index for 75/25 ratio
	total_files = left_images_count  
	split_index = int(total_files * 0.75)  # 75% of 800

	# split dset [left, right, disp] into train / validation folders
	split_to_train_validation(LEFT_IMAGES_FOLDER, TRAIN_LEFT_FOLDER, VALIDATION_LEFT_FOLDER, split_index)
	split_to_train_validation(RIGHT_IMAGES_FOLDER, TRAIN_RIGHT_FOLDER, VALIDATION_RIGHT_FOLDER, split_index)
	split_to_train_validation(LEFT_DISPARITY_FOLDER, TRAIN_DISPARITY_FOLDER,VALIDATION_DISPARITY_FOLDER, split_index)

	# generating .png disparity from .pfm files
	process_disp_folder(VALIDATION_DISPARITY_FOLDER)
	process_disp_folder(TRAIN_DISPARITY_FOLDER)

	# prepatre png-disparity files for finetune.py
	process_dset(VALIDATION_FOLDER)
	process_dset(TRAIN_FOLDER)

	
if __name__ == "__main__":
	coloredlogs.install(level="WARNING", force=True)  # install a handler on the root logger
	main()