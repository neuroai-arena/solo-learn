import argparse
import glob
import os
import subprocess
import time

import h5py
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/arc02/archive/autolearn/aubret/ego4d")
parser.add_argument("--range", type=int, default=95)
args=parser.parse_args()


hdf5_file = h5py.File(os.path.join(args.data_root, f"data_all{str(args.range)}.h5"), "w")
for p2 in range(args.range):
    if os.path.isfile(os.path.join(args.data_root, f"ego4d_{p2}.h5")):
        hdf5_file[f"/{str(p2)}"] = h5py.ExternalLink(f"{args.data_root}/ego4d_{p2}.h5", "/")
    else:
        raise Exception(f"{p2} should exist")
hdf5_file.close()

