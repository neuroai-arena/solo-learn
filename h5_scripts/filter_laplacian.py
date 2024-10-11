import argparse
import io
import os
import sys

from tqdm import tqdm
import cv2
import h5py
import numpy as np
import config

from util.getters import get_arguments
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/home/fias/postdoc/data/ego4d")
parser.add_argument("--split", type=int, default=10, help="Number of total chunks")
parser.add_argument("--start", type=int, default=0, help="Index of the chunk generated")
args=parser.parse_args()

data = h5py.File(os.path.join(args.data_root, f"data_all95.h5"), "r")
dataset = h5py.File(os.path.join(args.data_root, f"dataset_all95.h5"),"r").get("data")
data_size = len(dataset)
print("Array creation")

block_size = data_size//args.split #if args.start != args.split-1 else data_size//args.split +
end_block_size = block_size + data_size%args.split if args.start == args.split-1 else block_size

file_path = os.path.join(args.data_root, f"unblur_{args.start}_{args.split}.npy")
if os.path.exists(file_path):
    blur_store = np.load(file_path)

print("Total length:", data_size)
# unmatch= 0
for idx in tqdm(range(block_size*args.start, block_size*args.start + end_block_size)):
    r = dataset[idx]
    index, number, partition = int(r[6]), int(r[19]), str(int(r[5]))
    pil_img = Image.open(io.BytesIO(data.get(partition).get(f"images224_{str(number)}")[index]))
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F)
    blur_store[idx - block_size*args.start] = v.var() < 100
np.save(file_path, blur_store)

