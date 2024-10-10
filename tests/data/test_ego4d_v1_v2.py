import os.path
import sys
sys.path.append("../..")

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from solo.data.custom.ego4d import Ego4d
from solo.data.pretrain_dataloader import NCropAugmentation

t= NCropAugmentation( transforms.ToTensor(), 2)
dataset_v1 = Ego4d("/scratch/autolearn/aubret/ego4d/hdf5/", t, v2=False, gaze_size=224, time_window=0)
dataset_v2 = Ego4d("/scratch/autolearn/aubret/ego4d/hdf5_new/", t, v2=True, gaze_size=224, time_window=0)


dataloader_v1 = DataLoader(dataset_v1, shuffle=True, batch_size=1, num_workers=1)
dataloader_v2 = DataLoader(dataset_v2, shuffle=True, batch_size=1, num_workers=1)

torch.manual_seed(0)
itv1 = iter(dataloader_v1)
torch.manual_seed(0)
itv2 = iter(dataloader_v2)

if not os.path.exists("/scratch/autolearn/aubret/ego4d/samplesv1"):
    os.makedirs("/scratch/autolearn/aubret/ego4d/samplesv1")

if not os.path.exists("/scratch/autolearn/aubret/ego4d/samplesv2"):
    os.makedirs("/scratch/autolearn/aubret/ego4d/samplesv2")

for i, (i1, i2) in enumerate(zip(itv1, itv2)):
    torchvision.utils.save_image(i1[0][0], f"/scratch/autolearn/aubret/ego4d/samplesv1/{i}.png")
    torchvision.utils.save_image(i2[0][0], f"/scratch/autolearn/aubret/ego4d/samplesv2/{i}.png")
    if i > 30:
        break