import argparse
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/arc01/archive/autolearn/aubret/imgnet")
args=parser.parse_args()

hdf5_file = h5py.File(os.path.join(args.data_root, f"data2_train.h5"), "w")
for p in range(26):
    if os.path.isfile(os.path.join(args.data_root, f"data2_train_{str(p*50000)}.hdf5")):
        hdf5_file[f"/{str(p*50000)}"] = h5py.ExternalLink(f"{args.data_root}/data2_train_{str(p*50000)}.hdf5", "/")
    else:
        raise Exception(f"{str(p)} should exist")
hdf5_file.close()