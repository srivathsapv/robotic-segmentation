import os
from pathlib import Path

input_data_dir = "data/cropped_train"
target_data_dir = "data/train_test_split"
instrument_directories = ["instrument_dataset_%i"%i for i in range(1,9)]
instr_sub_directories = ["binary_masks", "images", "instruments_masks", "parts_masks"]

def link_file_range(src_dir, tgt_dir, file_ext, start_val, end_val):
    tgt_dir.mkdir(exist_ok=True)
    for i in range(start_val, end_val):
        frame_link = Path(tgt_dir / ("frame%03d.%s" % (i, file_ext)))
        img_file = (src_dir / ("frame%03d.%s"%(i,file_ext))).absolute()
        print(frame_link)
        frame_link.symlink_to(img_file)


for instrument_dir in instrument_directories:
    # E.g. instrument_dataset_1
    src_instr_dir = Path(input_data_dir+"/"+instrument_dir)

    new_train_instr_dir = Path(target_data_dir+"/train/"+instrument_dir)
    new_train_instr_dir.mkdir(parents=True, exist_ok=True)

    new_test_instr_dir = Path(target_data_dir+"/test/"+instrument_dir)
    new_test_instr_dir.mkdir(parents=True, exist_ok=True)

    for instr_sub_dir_name in instr_sub_directories:
        # E.g. binary_images
        src_instr_sub_dir = src_instr_dir / instr_sub_dir_name
        file_ext = "jpg" if  instr_sub_dir_name == "images" else "png"

        new_train_instr_sub_directory = new_train_instr_dir / instr_sub_dir_name
        link_file_range(src_instr_sub_dir, new_train_instr_sub_directory, file_ext, 0, 200)

        new_test_instr_sub_directory = new_test_instr_dir / instr_sub_dir_name
        link_file_range(src_instr_sub_dir, new_test_instr_sub_directory, file_ext, 200, 225)