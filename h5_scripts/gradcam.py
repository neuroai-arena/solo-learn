import json
import os

import numpy as np
import torch
import torchvision
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torchvision import transforms
from PIL import Image

# dataset_name = "cifar10"
# dataset_name = "stl10"
dataset_name = "imgnet"

# strmodels = 'cvm'
strmodels = 'ffm'

imgnet_mapping = json.load(open("/home/fias/postdoc/datasets/imgnet/imagenet_class_index.json","r"))

# ["Airliner", "Wagon", "Humming bird", "Siamese Cat", "Ox", "Golden Retriever","Tailed Frog", "Zebra"]

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

t = transforms.Compose(
            [
                transforms.Resize(224),# if dataset_name != "stl10" else 96),
                transforms.CenterCrop(224),# if dataset_name != "stl10" else 96),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
)

if dataset_name != "imgnet":
    normalize = transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261))
else:
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


if dataset_name in ["cifar10"]:
    d = CIFAR10(f"/home/fias/postdoc/datasets/cifar10", train=False, transform = t)
else:
    d = STL10(f"/home/fias/postdoc/datasets/stl10", split="test", transform = t)
os.makedirs(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{dataset_name}_{strmodels}/",exist_ok=True)


loader = DataLoader(d, batch_size=92, shuffle=False, pin_memory=True)
iterator = iter(loader)
rgb_images, target = next(iterator)
rgb_images, target = next(iterator)
images = normalize(rgb_images)

target_layers = [model.layer4[-1]]

print(torch.argmax(model(images),dim=1))
print(target)
outputs = torch.argmax(model(images), dim=1)
success = (outputs == target).to(torch.float32)
print("acc", success.mean())
with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
    for i in range(images.shape[0]):
        image = images[i:i+1]

        label = target[0]

        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = cam(input_tensor=image, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        # visualization = show_cam_on_image(image.cpu().numpy().transpose(1, 2, 0), grayscale_cam, use_rgb=True)
        visualization = show_cam_on_image(rgb_images[i].numpy().transpose(1, 2, 0), grayscale_cam, use_rgb=True, image_weight=0.7)

        if dataset_name == "imgnet":
            successim = imgnet_mapping[str(outputs[i].item())][1]
        else:
            successim = int(success[i].item())

        im = Image.fromarray(visualization)
        im.save(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{dataset_name}_{strmodels}/sal_{i}_{successim}.jpg")

        im2 = Image.fromarray(np.uint8(255 * rgb_images[i].numpy().transpose(1, 2, 0)))
        im2.save(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{dataset_name}_{strmodels}/{i}.jpg")

