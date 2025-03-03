import json
import os
import time
from typing import Optional, Callable, cast, Tuple, Any

import PIL
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset, CIFAR10
from torchvision.datasets.utils import verify_str_arg, check_integrity, download_and_extract_archive
from torchvision.models import resnet50
from torch import nn
from torchvision import transforms
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode


def main():
    # dataset_name = "cifar10"
    # dataset_name = "stl10"
    dataset_name = "imgnet"

    strmodels = 'ffm'
    # strmodels = 'cvm'

    setting = "all"
    black_mask = True

    imgnet_mapping = json.load(open("/home/fias/postdoc/datasets/imgnet/imagenet_class_index.json", "r"))
    imgnet_mapping2 = {c[0]: i for i, c in imgnet_mapping.items()}
    # print(imgnet_mapping2)
    # time.sleep(10)
    file = open("synsets-to-cifar10", "r")
    # categories = []
    i=-1
    mapping = {}
    mapping2 = {}
    cpt=0
    for r in file.readlines():
        if not r.startswith("-"):
            i+=1
            # categories.append({i : []})
        else:
            # categories[i].append(r.split(":")[0].split("-")[-1])
            try:
                ni = imgnet_mapping2[r.split(":")[0].split("-")[-1]]
                mapping[ni] = i
                mapping2[cpt] = i
                cpt += 1
            except Exception as e:
                pass
                # print(e)

    file.close()
    mappingk = torch.tensor(list(map(int, mapping.keys())), device="cuda:0").repeat(64,1)



    # imgnetclasses = ["airliner", "wagon", "humming_bird", "siamese_cat", "ox", "golden_retriever", "tailed_frog", "zebra",
    #  "container_ship", "trailer_truck"]
    # imgnetids = {imgnet_mapping[str(i)][1] :i for i in range(1000) if imgnet_mapping[str(i)][1] in imgnetclasses}
    # print(imgnet_mapping["0"])
    # print(imgnetids)
    # imgnetmapping = [imgnetids[ic] for ic in imgnetclasses]
    # allids = imgnetids.keys()

    # cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    central_vision_models = {
        'model_ckpt': 'iiiqnrs0/mocov3nf_lr16-iiiqnrs0-ep=last.ckpt',
        'imgnet': 'iiiqnrs0/linlast_mocov3nf_lr16_fullIN-sc7poe6z-ep=last.ckpt',
        'stl10': 'iiiqnrs0/v2_STL10_mocov3nf_lr16_lr1.0-45xzm2dt-ep=last-stp=last.ckpt',
        'cifar10': 'iiiqnrs0/v2_cifar10_224_mocov3nf_lr16_lr1.0-0ovrj9io-ep=last-stp=last.ckpt'
    }

    # time_window = 15, gaze_size = 540, center_crop = False, resize_gs = False
    full_frame_models = {
        'model_ckpt': 'ba1e2na0/mocov3nf_lr16_gs540-ba1e2na0-ep=last.ckpt',
        'imgnet': 'ba1e2na0/linlast_mocov3nf_lr16_gs540_fullIN-nn14c1wf-ep=last.ckpt',
        'stl10': 'ba1e2na0/v2_STL10_mocov3nf_lr16_gs540_lr1.0-2lm9md9t-ep=last-stp=last.ckpt',
        'cifar10': 'ba1e2na0/v2_cifar10_224_mocov3nf_lr16_gs540_lr1.0-6ja1gb2m-ep=last-stp=last.ckpt'
    }

    models = central_vision_models if strmodels == "cvm" else full_frame_models


    central_vision_models = torch.load(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{models['model_ckpt']}",map_location="cpu")["state_dict"]
    central_vision_linear = torch.load(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{models[dataset_name]}",map_location="cpu")["state_dict"]
    cvm = {}

    for k, v in central_vision_models.items():
        if "momentum" in k:
            continue
        if "backbone" in k:
            k2 = ".".join(k.split(".")[1:])
            cvm[k2] = v

    cvm["fc.weight"] = central_vision_linear["classifier.weight"]
    cvm["fc.bias"] = central_vision_linear["classifier.bias"]

    model = resnet50()
    model.fc = nn.Linear(2048, 10 if dataset_name != "imgnet" else 1000)
    model.load_state_dict(cvm)
    model.eval()
    model = model.to("cuda:0")

    t = transforms.Compose(
                [
                    # transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),# if dataset_name != "stl10" else 96),
                    transforms.Resize(224),# if dataset_name != "stl10" else 96),
                    # transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261))
                ]
    )

    normalize = transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261))

    if dataset_name in ["cifar10","cifar100","imgnet"]:
        d = CIFAR10(f"/home/fias/postdoc/datasets/cifar10", train=False, transform=t)
    else:
        d = STL10(f"/home/fias/postdoc/datasets/stl10", split="test", transform = t, setting=setting, black_mask=black_mask)
    loader = DataLoader(d, batch_size=64, shuffle=False, pin_memory=True)
    all_succ = 0
    all_cpt = 0
    with torch.no_grad():
        # for it, (images, target, id) in enumerate(loader):
        for it, (images, target) in enumerate(loader):
            if it < 10:
                torchvision.utils.save_image(images[0], f"/home/fias/postdoc/gym_results/test_images/stl10/test_mask{it}.png")
            images = normalize(images)
            out = model(images.to("cuda:0"))
            if dataset_name == "imgnet":
                try:
                    out = out.gather(1, mappingk)
                    outputs = torch.argmax(out, dim=1)
                    outputs_mapped = torch.tensor([mapping2[o.item()] for o in outputs])
                    success = (outputs_mapped == target).to(torch.float32)
                    all_succ += success.sum(dim=0)
                    all_cpt += images.shape[0]
                except Exception as e:
                    print(e)


            else:
                outputs = torch.argmax(out, dim=1)
                success = (outputs == target.to("cuda:0")).to(torch.float32)
                all_succ += success.sum(dim=0)
                all_cpt += images.shape[0]


            if it % 64*10 == 0:
                print(it, (all_succ/all_cpt).item())


    print(it, (all_succ/all_cpt).item())

if __name__ == "__main__":
    main()