import os.path
from os import path
import shutil
import argparse

parser = argparse.ArgumentParser(description='Move converted PNGs')
parser.add_argument('--data-dir', required=True,
                    help='data directory holding the resized images')
parser.add_argument('--output-dir', required=True,
                    help='data directory storing the resized images')
parser.add_argument('--file-identifier', default="_resized",
                    help='determines how to differentiate resized and original files')
args = parser.parse_args()

OUTPUT_TRAIN_DIR = os.path.join(args.output_dir, "train")
OUTPUT_TEST_DIR = os.path.join(args.output_dir, "test")

if not path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if not path.exists(OUTPUT_TRAIN_DIR):
    os.mkdir(OUTPUT_TRAIN_DIR)

if not path.exists(OUTPUT_TEST_DIR):
    os.mkdir(OUTPUT_TEST_DIR)

for path, subdirs, files in os.walk(args.data_dir):
    for name in files:
        if args.file_identifier in name:
            resized_file = os.path.join(path, name)
            # if "train" in path:
            #     copied_location = os.path.join(OUTPUT_TRAIN_DIR, "{}_{}".format(path[2:].replace(os.sep, '_'), name))
            # else:
            #     copied_location = os.path.join(OUTPUT_TEST_DIR, "{}_{}".format(path[2:].replace(os.sep, '_'), name))
            # print(copied_location)
            # shutil.copy(resized_file, copied_location)
            if "train" in name:
                copied_location = os.path.join(OUTPUT_TRAIN_DIR, "{}_{}".format(path[2:].replace(os.sep, '_'), name))
            else:
                copied_location = os.path.join(OUTPUT_TEST_DIR, "{}_{}".format(path[2:].replace(os.sep, '_'), name))
            print(copied_location)
            shutil.copy(resized_file, copied_location)

