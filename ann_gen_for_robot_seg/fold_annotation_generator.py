import json
from pathlib import Path
from shutil import copyfile

annotation_path = Path("../data/annotations")
input_images_path = Path("../data/cropped_train")

folds = {0: [1, 3],
         1: [2, 5],
         2: [4, 8],
         3: [6, 7]}

def add_intrument_id_to_images(instr_bin_anns, instrument_id):
    i = 0
    for image in instr_bin_anns["images"]:
        image['file_name'] = "ids_%i_%s"%(instrument_id, image["file_name"])
        image['instrument_id'] = instrument_id
        i += 1
    print("instrument_id: %i, n: %i"%(instrument_id, i))

for fold, val_instrument_ids in folds.items():
    fold_train_images = []
    fold_val_images = []

    fold_binary_train_annotations = []
    fold_binary_val_annotations = []

    fold_parts_train_annotations = []
    fold_parts_val_annotations = []

    for instrument_id in range(1, 9):
        instrument_json = 'instrument_dataset_' + str(instrument_id) + ".json"

        # Read input instrument-specific annotations
        instr_bin_json_path = annotation_path / 'binary' / instrument_json
        instr_bin_anns = json.load(open(str(instr_bin_json_path)))
        add_intrument_id_to_images(instr_bin_anns, instrument_id)

        instr_parts_json_path = annotation_path / 'parts' / instrument_json
        instr_parts_anns = json.load(open(str(instr_parts_json_path)))
        add_intrument_id_to_images(instr_parts_anns, instrument_id)

        # Add information of instrument to fold's train or eval annotations based on fold values
        if instrument_id in val_instrument_ids:
            print("fold:%i; id:%i ; type: val"%(fold, instrument_id))
            fold_val_images += instr_bin_anns["images"]

            fold_binary_val_annotations += instr_bin_anns["annotations"]
            fold_parts_val_annotations += instr_parts_anns["annotations"]
        else:
            print("fold:%i; id:%i ; type: train"%(fold, instrument_id))
            fold_train_images += instr_bin_anns["images"]

            fold_binary_train_annotations += instr_bin_anns["annotations"]
            fold_parts_train_annotations += instr_parts_anns["annotations"]


        # type: binary or parts

    def write_fold_mode_info(images, categories, annotations, type, train_eval):
        fold_path = annotation_path / ('%s_folds'%type) / ("fold_%i"%fold)

        target_image_dir = fold_path / ("%s2014"%train_eval)
        target_image_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            instrument_id = image['instrument_id']
            source_img_dir_path = input_images_path / ('instrument_dataset_%i' % instrument_id) / "images"
            source_img_path = str(source_img_dir_path / (image["file_name"].split("_")[2]))
            target_img_file_path = target_image_dir / image["file_name"]
            copyfile(str(source_img_path), str(target_img_file_path))

        consolidated_annotations = {'images': images, 'categories': categories, 'annotations': annotations}
        ann_path = fold_path / "annotations"
        ann_path.mkdir(exist_ok=True, parents=True)
        (ann_path / ("instances_%s2014.json" % (train_eval))).write_text(json.dumps(consolidated_annotations))

    write_fold_mode_info(fold_train_images, instr_bin_anns["categories"], fold_binary_train_annotations, 'binary', 'train')
    write_fold_mode_info(fold_val_images, instr_bin_anns["categories"], fold_binary_val_annotations, 'binary', 'val')
    write_fold_mode_info(fold_train_images, instr_parts_anns["categories"], fold_parts_train_annotations, 'parts', 'train')
    write_fold_mode_info(fold_val_images, instr_parts_anns["categories"], fold_parts_val_annotations, 'parts', 'val')
    print("data generated for fold %i"%fold)