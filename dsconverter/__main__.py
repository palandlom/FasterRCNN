import os
import sys

import dsconverter.conv

# dataset_path = "C:/Users/cride/Downloads/archive/"
# output_path = "../test/"
output_path = "./outDataset"

if len(sys.argv) < 2:
    print(f"Not enough args - input dataset path and optional output folder should")
    print("Use: python -m dsconverter ./input/dataset/ ./optional/output/folder/")
    os._exit(1)

dataset_path = sys.argv[1]

if len(sys.argv) < 3:
    print(f"Output folder was not set: {output_path} folder will be used")
    os.mkdir(output_path)
else:
    output_path = sys.argv[2]

dsconverter.conv.convert(dataset_path, output_path)
