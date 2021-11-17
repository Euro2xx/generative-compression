import os
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Create required dataframe and h5 file')
parser.add_argument('--data-dir', required=True,
                    help='data directory holding the resized images')
args = parser.parse_args()

pathList = []

for path, subdirs, files in os.walk(args.data_dir):
    for name in files:
        pathList.append(os.path.join(path, name))

data = {'path': pathList}

df = pd.DataFrame(data)
print(df.head())
#df.to_hdf("data\cifar.h5", key="df")


