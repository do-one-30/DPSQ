import os
import math
import pathlib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Any, Optional

from applications.custom_quant import CustomPointConv, CustomPointConvCalib, CustomPointConvEval

def replace_efficientvit_1x1_conv(model: nn.Module, method_func, tile_m, tile_n, tile_k, **kwargs):
    replaced_count = 0
    custom_modules = []
    for name, module in model.named_modules():
        if "local_module.main.point_conv.conv" in name:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            old_layer = getattr(parent_module, child_name)

            if isinstance(old_layer, nn.Conv2d) and old_layer.kernel_size == (1, 1):
                new_layer = CustomPointConv(
                    in_channels=old_layer.in_channels, out_channels=old_layer.out_channels,
                    method_func=method_func, bias=(old_layer.bias is not None),
                    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, **kwargs
                ).to(old_layer.weight.device, dtype=old_layer.weight.dtype)

                with torch.no_grad():
                    new_layer.weight.copy_(old_layer.weight.squeeze())
                    if old_layer.bias is not None: 
                        new_layer.bias.copy_(old_layer.bias)

                setattr(parent_module, child_name, new_layer)
                custom_modules.append(new_layer)
                replaced_count += 1
                
    return model, custom_modules


def replace_efficientvit_1x1_conv_for_calib(model: nn.Module, calib_method, tile_size, gs, eps):
    replaced_count = 0
    for name, module in model.named_modules():
        if "local_module.main.point_conv.conv" in name:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            old_layer = getattr(parent_module, child_name)

            if isinstance(old_layer, nn.Conv2d) and old_layer.kernel_size == (1, 1):
                new_layer = CustomPointConvCalib(
                    in_channels=old_layer.in_channels, out_channels=old_layer.out_channels,
                    calib_method=calib_method, bias=(old_layer.bias is not None),
                    tile_m=tile_size, tile_n=tile_size, tile_k=tile_size, gs=gs, eps=eps
                ).to(old_layer.weight.device, dtype=old_layer.weight.dtype)

                with torch.no_grad():
                    new_layer.linear_layer.weight.copy_(old_layer.weight.squeeze())
                    if old_layer.bias is not None: new_layer.linear_layer.bias.copy_(old_layer.bias)

                setattr(parent_module, child_name, new_layer)
                replaced_count += 1
    return model


def replace_efficientvit_1x1_conv_for_eval(model: nn.Module, apply_method, layer_scales, tile_size, gs, eps):
    replaced_count = 0
    for name, module in model.named_modules():
        if "local_module.main.point_conv.conv" in name:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            old_layer = getattr(parent_module, child_name)

            if isinstance(old_layer, nn.Conv2d) and old_layer.kernel_size == (1, 1):
                if name not in layer_scales: continue
                new_layer = CustomPointConvEval(
                    in_channels=old_layer.in_channels, out_channels=old_layer.out_channels,
                    apply_method=apply_method, step_scales=layer_scales[name], bias=(old_layer.bias is not None),
                    tile_m=tile_size, tile_n=tile_size, tile_k=tile_size, gs=gs, eps=eps
                ).to(old_layer.weight.device, dtype=old_layer.weight.dtype)

                with torch.no_grad():
                    new_layer.linear_layer.weight.copy_(old_layer.weight.squeeze())
                    if old_layer.bias is not None: new_layer.linear_layer.bias.copy_(old_layer.bias)

                setattr(parent_module, child_name, new_layer)
                replaced_count += 1
    return model

class Resize(object):
    def __init__(
        self,
        crop_size: Optional[tuple[int, int]],
        interpolation: Optional[int] = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.crop_size is None or self.interpolation is None:
            return feed_dict

        image, target = feed_dict["data"], feed_dict["label"]
        height, width = self.crop_size

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
        return {
            "data": image,
            "label": target,
        }


class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        image, mask = feed_dict["data"], feed_dict["label"]
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {
            "data": image,
            "label": mask,
        }


class SegIOU:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }


class CityscapesDataset(Dataset):
    classes = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
    class_colors = (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    )
    label_map = np.array(
        (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,  # road 7
            1,  # sidewalk 8
            -1,
            -1,
            2,  # building 11
            3,  # wall 12
            4,  # fence 13
            -1,
            -1,
            -1,
            5,  # pole 17
            -1,
            6,  # traffic light 19
            7,  # traffic sign 20
            8,  # vegetation 21
            9,  # terrain 22
            10,  # sky 23
            11,  # person 24
            12,  # rider 25
            13,  # car 26
            14,  # truck 27
            15,  # bus 28
            -1,
            -1,
            16,  # train 31
            17,  # motorcycle 32
            18,  # bicycle 33
        )
    )

    def __init__(self, data_dir: str, crop_size: Optional[tuple[int, int]] = None):
        super().__init__()

        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".png"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace(
                    "_leftImg8bit.", "_gtFine_labelIds."
                )
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        # build transform
        self.transform = transforms.Compose(
            [
                Resize(crop_size),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.label_map[mask]

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }


class ADE20KDataset(Dataset):
    classes = (
        "wall",
        "building",
        "sky",
        "floor",
        "tree",
        "ceiling",
        "road",
        "bed",
        "windowpane",
        "grass",
        "cabinet",
        "sidewalk",
        "person",
        "earth",
        "door",
        "table",
        "mountain",
        "plant",
        "curtain",
        "chair",
        "car",
        "water",
        "painting",
        "sofa",
        "shelf",
        "house",
        "sea",
        "mirror",
        "rug",
        "field",
        "armchair",
        "seat",
        "fence",
        "desk",
        "rock",
        "wardrobe",
        "lamp",
        "bathtub",
        "railing",
        "cushion",
        "base",
        "box",
        "column",
        "signboard",
        "chest of drawers",
        "counter",
        "sand",
        "sink",
        "skyscraper",
        "fireplace",
        "refrigerator",
        "grandstand",
        "path",
        "stairs",
        "runway",
        "case",
        "pool table",
        "pillow",
        "screen door",
        "stairway",
        "river",
        "bridge",
        "bookcase",
        "blind",
        "coffee table",
        "toilet",
        "flower",
        "book",
        "hill",
        "bench",
        "countertop",
        "stove",
        "palm",
        "kitchen island",
        "computer",
        "swivel chair",
        "boat",
        "bar",
        "arcade machine",
        "hovel",
        "bus",
        "towel",
        "light",
        "truck",
        "tower",
        "chandelier",
        "awning",
        "streetlight",
        "booth",
        "television receiver",
        "airplane",
        "dirt track",
        "apparel",
        "pole",
        "land",
        "bannister",
        "escalator",
        "ottoman",
        "bottle",
        "buffet",
        "poster",
        "stage",
        "van",
        "ship",
        "fountain",
        "conveyer belt",
        "canopy",
        "washer",
        "plaything",
        "swimming pool",
        "stool",
        "barrel",
        "basket",
        "waterfall",
        "tent",
        "bag",
        "minibike",
        "cradle",
        "oven",
        "ball",
        "food",
        "step",
        "tank",
        "trade name",
        "microwave",
        "pot",
        "animal",
        "bicycle",
        "lake",
        "dishwasher",
        "screen",
        "blanket",
        "sculpture",
        "hood",
        "sconce",
        "vase",
        "traffic light",
        "tray",
        "ashcan",
        "fan",
        "pier",
        "crt screen",
        "plate",
        "monitor",
        "bulletin board",
        "shower",
        "radiator",
        "glass",
        "clock",
        "flag",
    )
    class_colors = (
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    )

    def __init__(self, data_dir: str, crop_size=512):
        super().__init__()

        self.crop_size = crop_size
        # load samples
        samples = []
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".jpg"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/images/", "/annotations/")
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        self.transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.int64) - 1

        h, w = image.shape[:2]
        if h < w:
            th = self.crop_size
            tw = math.ceil(w / h * th / 32) * 32
        else:
            tw = self.crop_size
            th = math.ceil(h / w * tw / 32) * 32
        if th != h or tw != w:
            image = cv2.resize(
                image,
                dsize=(tw, th),
                interpolation=cv2.INTER_CUBIC,
            )

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple | list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas