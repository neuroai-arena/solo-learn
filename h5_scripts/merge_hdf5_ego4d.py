import argparse
import glob
import os
import subprocess
import time

from tqdm import trange
import h5py
import numpy as np
import pandas as pd

def str2table(v):
    return v.split(',')

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/scratch/modelrep/schaumloeffel/Ego4d/")
parser.add_argument("--range", type=int, default=95)
# parser.add_argument("--omit", type=str2table, default=[71,56,67,74])
parser.add_argument("--omit", type=str2table, default=[])
parser.add_argument("--nl", type=int, default=0)
args=parser.parse_args()
print(args.omit)


filename = os.path.join(args.data_root, f"dataset_all{args.range}v2.h5")

types = {i: "float32" for i in range(6,12)}
types.update({0: "string", 1: "int32", 2: "int32"})
types.update({3: "int32", 4: "int32", 5: "int32"})
nl = args.nl
lengths = []
if args.nl == 0:
    for p in trange(args.range):
        if p in args.omit:
            continue
        v = int(subprocess.check_output("wc -l "+os.path.join(args.data_root, f"dataset{str(p)}.csv"), shell=True).split()[0])
        nl += v
        lengths.append(v)
print("total nl", nl)
f = h5py.File(filename, "w")
data = f.create_dataset("data", (nl,12))
pos = 0
csvfs = []
for p in trange(args.range):
    if p in args.omit:
        continue
    p2=str(p)
    try:
        f = os.path.join(args.data_root, f"dataset{str(p2)}.csv")
        csvf = pd.read_csv(f, header=None, dtype=types)
        size = len(csvf)
    except Exception as e:
        print(f)
        raise Exception(e)
    if size != lengths[p]:
        print(size, lengths[p], p)

    csvf[0] = csvf[0].apply(lambda x: hash(x))
    csvf = csvf.sort_values(by=[0,1],axis=0)
    data[pos:pos+len(csvf)] = csvf.values
    pos += size



hdf5_file = h5py.File(os.path.join(args.data_root, f"data_all{str(args.range)}v2.h5"), "w")
for p2 in range(args.range):
    if p in args.omit:
        continue
    if os.path.isfile(os.path.join(args.data_root, f"data{p2}.hdf5")):
        hdf5_file[f"/{str(p2)}"] = h5py.ExternalLink(f"{args.data_root}/data{p2}.hdf5", "/")
    else:
        raise Exception(f"{p2} should exist")
hdf5_file.close()

