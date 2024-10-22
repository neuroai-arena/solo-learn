import argparse
import os

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/scratch/autolearn/aubret/ego4dv2")
parser.add_argument("--range", type=int, default=95)
args=parser.parse_args()


dataset = h5py.File(os.path.join(args.data_root, f"dataset_all95.h5"), "r")["data"]
hdf5_file = h5py.File(os.path.join(args.data_root, f"data_all{str(args.range)}.h5"), "r")

dataset = np.concatenate((dataset[:,5:6,],dataset[:,11:12]), axis=1)
values = np.unique(dataset, axis=0)

for i in range(len(values)):
    partition = str(int(values[i,0]))
    chunk = str(int(values[i,1]))
    try:
        d = hdf5_file[partition]["frames"][f"images540_{chunk}"]
    except:
        print("Error:", partition, chunk, flush=True)

print(values)

#
# for p in range(args.range):
#     partition = hdf5_file[p]
#     try:
#         print()
#     if p in args.omit:
#         continue
#     if os.path.isfile(os.path.join(args.data_root, f"data{p2}.hdf5")):
#         hdf5_file[f"/{str(p2)}"] = h5py.ExternalLink(f"{args.data_root}/data{p2}.hdf5", "/")
#     else:
#         raise Exception(f"{p2} should exist")
