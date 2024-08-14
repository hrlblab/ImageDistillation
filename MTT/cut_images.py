import os
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, current_process
def slice_image(image, size=32):
    """Slice the image into size x size patches."""
    patches = []
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            patch = image[i:i + size, j:j + size]
            # Check if the patch size is 128x128
            if patch.shape[0] == size and patch.shape[1] == size:
                patches.append(patch)
    return patches

def is_mostly_black(image, threshold=0.2):
    """Check if the image has a black part over the given threshold."""
    total_pixels = np.prod(image.shape)
    black_pixels = np.sum(image == 0)
    if black_pixels / total_pixels > threshold:
        return True
    return False

def process_image(filename, original_folder, mask_folder, output_folder):
    original_img_path = os.path.join(original_folder, filename)
    mask_img_path = os.path.join(mask_folder, filename.replace('.png', '_mask.png'))

    original_image = cv2.imread(original_img_path)
    mask_image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    original_patches = slice_image(original_image)
    mask_patches = slice_image(mask_image)

    for idx, mask_patch in enumerate(mask_patches):
        if not is_mostly_black(mask_patch):
            patch_name = f"{filename.replace('.png', '')}_patch_{idx}.png"
            patch_path = os.path.join(output_folder, patch_name)
            cv2.imwrite(patch_path, original_patches[idx])

    return filename  # Return processed filename for tqdm's update
if __name__ == "__main__":
    base_path = '/run/user/1443748932/gvfs/smb-share:server=hrlblab-nas2.it.vanderbilt.edu,share=data/lim47/Data Distillation/800_Modified/'
    folders = ['HP', 'NORM', 'TA.HG', 'TA.LG', 'TVA.HG', 'TVA.LG']
    total_files = sum([len(os.listdir(os.path.join(base_path, folder))) for folder in folders])
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for folder in folders:
            original_folder = os.path.join(base_path, folder)
            mask_folder = os.path.join(base_path, folder + "_mask")
            output_folder = os.path.join(base_path, folder + "_32")

            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            filenames = [f for f in os.listdir(original_folder) if f.endswith('.png') and "_mask.png" not in f]
            for filename in filenames:
                original_img_path = os.path.join(original_folder, filename)
                mask_img_path = os.path.join(mask_folder, filename.replace('.png', '_mask.png'))

                original_image = cv2.imread(original_img_path)
                mask_image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

                original_patches = slice_image(original_image)
                mask_patches = slice_image(mask_image)

                for idx, mask_patch in enumerate(mask_patches):
                    if not is_mostly_black(mask_patch):
                        patch_name = f"{filename.replace('.png', '')}_patch_{idx}.png"
                        patch_path = os.path.join(output_folder, patch_name)
                        cv2.imwrite(patch_path, original_patches[idx])

                pbar.update(1)