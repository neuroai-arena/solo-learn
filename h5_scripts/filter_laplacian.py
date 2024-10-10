import argparse
import io
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import cv2
import h5py
import numpy as np
import config

from util.getters import get_arguments
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/home/fias/postdoc/data/ego4d")
parser.add_argument("--range", type=int, default=10)
parser.add_argument("--start", type=int, default=0)
args=parser.parse_args()

data = h5py.File(os.path.join(args.data_root, f"data_all95.h5"), "r")
dataset = h5py.File(os.path.join(args.data_root, f"dataset_all95.h5"),"r").get("data")
data_size = len(dataset)
print("Array creation")

block_size = data_size//args.range #if args.start != args.range-1 else data_size//args.range +
end_block_size = block_size + data_size%args.range if args.start == args.range-1 else block_size

blur_store = np.load(os.path.join(args.data_root, f"unblur_{args.start}_{args.range}.npy"))
print("Total length:", data_size)
unmatch= 0
start_i= block_size*args.start + end_block_size - 100000
for idx in tqdm(range( start_i, block_size*args.start + end_block_size)):
    r = dataset[idx]
    index, number, partition = int(r[6]), int(r[19]), str(int(r[5]))
    pil_img = Image.open(io.BytesIO(data.get(partition).get(f"images224_{str(number)}")[index]))
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F)
    blur = v.var() < 100
    if blur and not blur_store[idx - block_size*args.start]:
        unmatch += 1
    blur_store[idx - block_size*args.start] = blur
    if idx%10000:
        print(idx, unmatch)
print(idx, unmatch)
np.save(os.path.join(args.data_root, f"unblur_{args.start}_{args.range}.npy"), blur_store)

