from pathlib import Path
import json
import argparse
import configparser
import os
import pandas as pd

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process some data paths.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    name_map = {}

    image_dirs = dataset_path / "images"
    image_lst = os.listdir(image_dirs)
    image_lst.sort()
    test_list = [img for img in image_lst if 'IMG_0499.jpg' <= img]
    img_root = str(image_dirs) + "/"
    img_file_type = image_lst[0].split(".")[-1]

    sd_dirs = dataset_path / "SD"
    sd_root = str(sd_dirs) + "/"
    sd_list = os.listdir(sd_dirs)
    sd_file_type = sd_list[0].split('.')[-1]


    for i in range(len(image_lst)):
        sd_file_name = image_lst[i].split('.')[0]+'.'+sd_file_type
        image_name = image_lst[i].split('.')[0]
        if not (os.path.exists(os.path.join(sd_root, sd_file_name))): continue
        if image_lst[i] not in test_list:
            os.rename(
                Path(img_root + image_lst[i]),
                Path(img_root + "2clutter" + image_name + "." + img_file_type),
            )
            os.rename(
                Path(sd_root + sd_file_name),
                Path(sd_root + "2clutter" + image_name + "." + sd_file_type),
            )
            name_map[image_lst[i]] = "2clutter" + image_name + "." + img_file_type

        elif image_lst[i] in test_list:
            os.rename(
                Path(img_root + image_lst[i]),
                Path(img_root + "1extra" + image_name + "." + img_file_type),
            )
            os.rename(
                Path(sd_root + sd_file_name),
                Path(sd_root + "1extra" + image_name + "." + sd_file_type),
            )
            name_map[image_lst[i]] = "1extra" + image_name + "." + img_file_type
    with open(os.path.join(dataset_path, 'name_map.json'), 'w') as f:
        json.dump(name_map, f)


if __name__ == "__main__":
    main()