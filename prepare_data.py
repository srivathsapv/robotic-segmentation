"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
from prop import fx, fy

data_path = Path('data')

train_path = data_path / 'train'

cropped_train_path = data_path / 'cropped_train'

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

height, width, h_start, w_start, original_height, original_width = int(height*fy), int(width*fx), int(h_start*fy), int(w_start*fx), int(original_height*fy), int(original_width*fx)

binary_factor = 255
parts_factor = 85  # 255/3
instrument_factor = 32


def read_im(file_name, is_mask=False):
    from prop import fx, fy
    if is_mask:
        img = cv2.imread(str(file_name), 0)
        # For masks, INTER_NEAREST is used instead of INTER_AREA (the typical choisce for shrinking) because the mask value is an indicator of part of instrument
        img = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.imread(str(file_name))
        img = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    return img


if __name__ == '__main__':
    for instrument_index in range(1, 9):
        instrument_folder = 'instrument_dataset_' + str(instrument_index)

        (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)

        binary_mask_folder = (cropped_train_path / instrument_folder / 'binary_masks')
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = (cropped_train_path / instrument_folder / 'parts_masks')
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = (cropped_train_path / instrument_folder / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        mask_folders = list((train_path / instrument_folder / 'ground_truth').glob('*'))
        # mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]

        for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
            img = read_im(file_name)
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

            mask_binary = np.zeros((old_h, old_w))
            mask_parts = np.zeros((old_h, old_w))
            mask_instruments = np.zeros((old_h, old_w))

            for mask_folder in mask_folders:
                if "DS_Store" in str(mask_folder): continue
                mask = read_im(mask_folder / file_name.name, is_mask=True)
                #mask = np.uint8(np.round(mask / 10)) * 10

                if 'Bipolar_Forceps' in str(mask_folder):
                    mask_instruments[mask > 0] = 1
                elif 'Prograsp_Forceps' in str(mask_folder):
                    mask_instruments[mask > 0] = 2
                elif 'Large_Needle_Driver' in str(mask_folder):
                    mask_instruments[mask > 0] = 3
                elif 'Vessel_Sealer' in str(mask_folder):
                    mask_instruments[mask > 0] = 4
                elif 'Grasping_Retractor' in str(mask_folder):
                    mask_instruments[mask > 0] = 5
                elif 'Monopolar_Curved_Scissors' in str(mask_folder):
                    mask_instruments[mask > 0] = 6
                elif 'Other' in str(mask_folder):
                    mask_instruments[mask > 0] = 7

                if 'Other' not in str(mask_folder):
                    mask_binary += mask

                    mask_parts[mask == 10] = 1  # Shaft
                    mask_parts[mask == 20] = 2  # Wrist
                    mask_parts[mask == 30] = 3  # Claspers
                    #cv2.imshow("mask",mask)
                    #cv2.waitKey(0)

            mask_binary = (mask_binary[h_start: h_start + height, w_start: w_start + width] > 0).astype(
                np.uint8) * binary_factor
            mask_parts = (mask_parts[h_start: h_start + height, w_start: w_start + width]).astype(
                np.uint8) * parts_factor
            mask_instruments = (mask_instruments[h_start: h_start + height, w_start: w_start + width]).astype(
                np.uint8) * instrument_factor

            cv2.imshow("mask_binary",mask_binary); cv2.waitKey(0)
            cv2.imshow("mask_parts", mask_parts); cv2.waitKey(0)
            cv2.imshow("mask_instruments", mask_instruments); cv2.waitKey(0)

            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)

def get_factor_mask_labels(problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
        labels = ['background', 'foreground']
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
        labels = ['background','Shaft','Wrist','Claspers']
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments_masks'
        labels = [] # TODO implement this
    return factor, mask_folder, labels