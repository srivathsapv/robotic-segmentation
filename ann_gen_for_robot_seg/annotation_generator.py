import json
from pytorch_mask_rcnn.pycocotools import mask as mask_tools
from skimage import measure
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
from prop import fx, fy
import os

data_path = Path('../data')

train_path = data_path / 'train'

mask_train_path = data_path / 'mask_rcnn_train'

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

# TODO modify fx, fy to ensure new dimensions are multiples of 64
height, width, h_start, w_start, original_height, original_width = int(height*fy), int(width*fx), int(h_start*fy), int(w_start*fx), int(original_height*fy), int(original_width*fx)

binary_factor = 255
parts_factor = 85  # 255/3
instrument_factor = 32


def read_im(file_name, is_mask=False):
    from prop import fx, fy
    # TODO modify fx, fy to ensure new dimensions are multiples of 64
    if is_mask:
        img = cv2.imread(str(file_name), 0)
        # For masks, INTER_NEAREST is used instead of INTER_AREA (the typical choisce for shrinking) because the mask value is an indicator of part of instrument
        img = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.imread(str(file_name))
        img = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    return img

# https://github.com/cocodataset/cocoapi/issues/131
def get_annotation_from_mask(ground_truth_binary_mask, image_id, category_id, id):
    # dtype=np.uint8

    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask_tools.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask_tools.area(encoded_ground_truth)
    ground_truth_bounding_box = mask_tools.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": ground_truth_bounding_box.tolist(),
        "category_id": category_id,
        "id": id
    }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    return annotation
    # print(json.dumps(annotation, indent=4))

if __name__ == '__main__':
    generate_cropped_images = False
    debug_images = True

    # Value of dict contains name and pixel-value
    part_dict = {1:('Shaft', 10), 2:('Wrist', 20), 3:('Claspers', 30)}

    bin_categories = [{'id': 1, 'name': 'instrument'}]
    parts_categories = [{'id':id, 'name': val[0]} for id,val in part_dict.items()]

    image_id = 0
    annotation_id = 0
    for instrument_index in range(1, 9):
        images = []

        binary_annotations = []
        parts_annotations = []

        instrument_folder = 'instrument_dataset_' + str(instrument_index)
        if generate_cropped_images:
            (mask_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)

        """
        binary_mask_folder = (mask_train_path / instrument_folder / 'binary_masks')
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = (mask_train_path / instrument_folder / 'parts_masks')
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = (mask_train_path / instrument_folder / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)"""

        mask_folders = list((train_path / instrument_folder / 'ground_truth').glob('*'))
        # mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]

        for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
            img = read_im(file_name)
            old_h, old_w, _ = img.shape

            # Not included license, coco_url, date_captured, flickr_url
            img_annotation = {'file_name': os.path.basename(file_name), 'id': image_id, 'height': height, 'width': width}
            images.append(img_annotation)

            # Crop required region
            img = img[h_start: h_start + height, w_start: w_start + width]
            if debug_images:
                cv2.imshow("img", img);
                cv2.waitKey(0)
            if generate_cropped_images:
                cv2.imwrite(str(mask_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

            # mask_binary = np.zeros((old_h, old_w))
            # mask_parts = np.zeros((old_h, old_w))
            # mask_instruments = np.zeros((old_h, old_w))

            for mask_folder in mask_folders:
                if "DS_Store" in str(mask_folder): continue
                mask = read_im(mask_folder / file_name.name, is_mask=True)
                mask = mask[h_start: h_start + height, w_start: w_start + width].astype(np.uint32)

                if np.all(mask == 0): continue
                #mask = np.uint8(np.round(mask / 10)) * 10
                """
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
                """
                if 'Other' not in str(mask_folder):
                    binary_mask = (mask > 0).astype(np.uint8)
                    bin_img_annotation = get_annotation_from_mask(binary_mask, image_id, 1, annotation_id)
                    if debug_images:
                        cv2.imshow("binary_mask", binary_mask*255);
                        cv2.waitKey(0)
                    binary_annotations.append(bin_img_annotation)
                    annotation_id += 1

                    for id, part_pixelVal in part_dict.items():
                        if np.any(mask == part_pixelVal[1]):
                            part_mask = (mask == part_pixelVal[1]).astype(np.uint8)
                            if debug_images:
                                cv2.imshow("part_mask", part_mask*255);
                                cv2.waitKey(0)
                            part_img_annotation = get_annotation_from_mask(part_mask, image_id, id, annotation_id)
                            parts_annotations.append(part_img_annotation)
                            annotation_id += 1

            """
            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
            """

            image_id += 1

        bin_consolidated_annotations = {'images': images, 'categories': bin_categories, 'annotations': binary_annotations}
        parts_consolidated_annotations = {'images': images, 'categories': parts_annotations, 'annotations': binary_annotations}
        print(1)


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

