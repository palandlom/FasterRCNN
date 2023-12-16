import os
import shutil
import sys

import dsconverter.conv

# TODO временный файл для запуска конвертера с произвольно папкой

# dataset_path = "C:/Users/cride/Downloads/archive/"
dataset_path = "/drv/necLindata/ml/NeuroBirds/"
# output_path = "../test/"
output_path = "./outDatasetNeuroBirds"

shutil.rmtree(output_path)

os.mkdir(output_path)

# "train", "test"
dsconverter.conv.convert(dataset_path, output_path)


