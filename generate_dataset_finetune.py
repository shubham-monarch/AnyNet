#! /usr/bin/env python3

import os
import shutil
from glob import glob
import logging, coloredlogs


from . import utils_anynet
from dataloader import readpfm

# Define source folders
DATASET_FOLDER = "dataset-anynet"
LEFT_IMAGES_FOLDER = f"{DATASET_FOLDER}/image_2"
RIGHT_IMAGES_FOLDER = f"{DATASET_FOLDER}/image_3"
LEFT_DISPARITY_FOLDER = f"{DATASET_FOLDER}/disp_occ_0"

# Define destination folders
DATASET_FOLDER_FINETUNE = "dataset-finetune"
TRAIN_FOLDER = f"{DATASET_FOLDER_FINETUNE}/training"
VALIDATION_FOLDER = f"{DATASET_FOLDER_FINETUNE}/validation"

# Folders to create
FOLDERS_TO_CREATE = [DATASET_FOLDER_FINETUNE]


# Function to split and copy files
def split_to_train_validation(source_folder, dest_train_folder, dest_validation_folder, split_index):
	files = sorted(glob(f"{source_folder}/*"))
	train_files = files[:split_index]
	validation_files = files[split_index:]

	os.makedirs(dest_train_folder, exist_ok=True)
	os.makedirs(dest_validation_folder, exist_ok=True)

	for file in train_files:
		shutil.copy(file, dest_train_folder)
		
	for file in validation_files:
		shutil.copy(file, dest_validation_folder)


def process_disp_folder(source_folder, dest_folder = None):
	files = sorted(glob(f"{source_folder}/*"))
	
	if dest_folder is None:
		dest_folder = f"{source_folder}-png"
		os.makedirs(dest_folder, exist_ok=True)

	for file in files:
		# Read the PFM file
		with open(file, "rb") as f:
			image, scale = utils_anynet.load_pfm(f)

		# Save the PFM file
		with open(f"{dest_folder}/{os.path.basename(file)}", "wb") as f:
			utils_anynet.save_pfm(f, image, scale)

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

	# Split and copy files for each folder
	split_to_train_validation(LEFT_IMAGES_FOLDER, f"{TRAIN_FOLDER}/image_2/", f"{VALIDATION_FOLDER}/image_2/", split_index)
	split_to_train_validation(RIGHT_IMAGES_FOLDER, f"{TRAIN_FOLDER}/image_3/", f"{VALIDATION_FOLDER}/image_3/", split_index)
	split_to_train_validation(LEFT_DISPARITY_FOLDER, f"{TRAIN_FOLDER}/disp_occ_0/", f"{VALIDATION_FOLDER}/disp_occ_0/", split_index)


if __name__ == "__main__":
	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	main()