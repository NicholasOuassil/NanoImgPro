import sys
import os
from pathlib import Path
from tqdm import tqdm
from NanoImgPro.NanoImgPro import NanoImgPro


folder_path = '/Documents/new_image_processing_data'  # Fill in path of data
p = Path(folder_path)
tiff_files = list(p.glob("*/*/*.tif")) # adjust for location of data

with tqdm(total=len(tiff_files)) as pbar:
  for filename in tiff_files:
    nano_img_pro = NanoImgPro(str(filename), 15)
    nano_img_pro.process_file(loading_bar=False)
    nano_img_pro.save_data()
    pbar.update(1)