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



raw_name, ext = args.name.split(".")[0],args.name.split(".")[1]
filename = os.path.join(args.data_root, f"{raw_name}.h5")
#https://datascience.stackexchange.com/questions/53125/file-converter-from-csv-to-hdf5

types = {i: "float32" for i in range(6,19)}
types.update({0: "string", 1: "int32", 2: "int32"})
types.update({3: "int32", 4: "int32", 5: "int32", 19: "int32"})
nl = 0
for p in range(args.range):
    nl += int(subprocess.check_output("wc -l "+os.path.join(args.data_root, f"dataset{str(p)}.csv"), shell=True).split()[0])
    print(nl)
print("total nl", nl)
f = h5py.File(filename, "w")
data = f.create_dataset("data", (nl,20))
pos = 0
csvfs = []
for p in range(args.range):
    p2=str(p)
    f = os.path.join(args.data_root, f"dataset{str(p2)}.csv")
    csvf = pd.read_csv(f, header=None, dtype=types)
    csvf[0] = csvf[0].apply(lambda x: hash(x))
    csvf = csvf.sort_values(by=[0,1],axis=0)
    data[pos:pos+len(csvf)] = csvf.values
    pos += len(csvf)





hdf5_file = h5py.File(os.path.join(args.data_root, f"data_all{str(args.range)}.h5"), "w")
for p2 in range(args.range):
    if os.path.isfile(os.path.join(args.data_root, f"ego4d_{p2}.h5")):
        hdf5_file[f"/{str(p2)}"] = h5py.ExternalLink(f"{args.data_root}/ego4d_{p2}.h5", "/")
    else:
        raise Exception(f"{p2} should exist")
hdf5_file.close()

