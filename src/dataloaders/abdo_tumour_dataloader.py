import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchio as tio

import SimpleITK as sitk
import json

from typing import List, Tuple, Dict
import sys

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

def build_dataloader(
        img_size: int = 512, 
        indices: List[int] = [4, 27, 26, 22, 6, 7],
        normalize_slices: bool = True,
        axes: str = ["axial"],
        window_level: List[int] = [400, 50],
        training: bool = False,
        data_dict_path: str = "/vol/biomedic3/bglocker/msc2023/cw1422/code/my_dataloaders/abdo_tumours/data_dict.json",
        mean: float = None,
        std: float = None
) -> Tuple[List, List, List]:
    
    with open(data_dict_path, "r") as f:
        data_dict = json.load(f)

    data_dict = {key: data_dict[key] for key in data_dict if int(key) in indices}
    dataset = AbdoDataset(
        data_dict=data_dict, 
        normalize_slices=normalize_slices,
        axes=axes,
        img_size=img_size,
        window_level=window_level,
        training=training,
        mean=mean,
        std=std
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


class AbdoDataset(Dataset):
    def __init__(
            self, 
            data_dict: str,
            normalize_slices: bool,
            axes: str,
            img_size: int,
            window_level: List[int],
            training: bool,
            mean: float,
            std: float
    ) -> None:
        self.data_dict = data_dict
        self.normalize_slices = normalize_slices
        self.window_level = window_level
        self.img_size = img_size
        self.training = training

        self.mean = mean
        self.std = std

        if axes == "all":
            self.axes = ["axial", "coronal", "sagittal"]
        else:
            self.axes = axes
        
        self.nums = [[], []]
        self.image_resizer = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.load_all_stacks()
        self.build_flattened_dataset()

    def wl_to_lh(self, window, level):
        low = level - window/2
        high = level + window/2
        return low, high

    def load_image_stack(
            self, 
            fname: str
    ) -> torch.Tensor:
        stack = sitk.ReadImage(fname)
        stack = sitk.GetArrayFromImage(stack)
        stack = torch.from_numpy(stack)
        stack = torch.swapaxes(stack, 0, 1)
        stack = transforms.functional.resize(
            stack, 
            (self.img_size, self.img_size),
            antialias=True
            )
        stack = torch.swapaxes(stack, 0, 1)

        if self.window_level != [0, 0]:
            low, high = self.wl_to_lh(self.window_level[0], self.window_level[1])
            stack = torch.clip(stack, low, high)
        
        normaliser = tio.RescaleIntensity(
            out_min_max=(0, 255), 
            in_min_max=(torch.min(stack).item(), torch.max(stack).item())
            )
        stack = normaliser(stack[None, :, : :])
        return stack
    
    def load_label_stack(
            self, 
            fname: str,
            class_id: int
    ) -> torch.Tensor:
        stack = sitk.ReadImage(fname)
        stack = sitk.GetArrayFromImage(stack)
        stack = torch.from_numpy(stack)
        stack = torch.swapaxes(stack, 0, 1)

        stack[stack != class_id] = 0
        stack[stack == class_id] = 1

        stack = transforms.functional.resize(
            stack, 
            (self.img_size, self.img_size),
            antialias=True
            )
        
        stack[stack > 0] = 1
        stack = torch.swapaxes(stack, 0, 1)

        stack = stack[None, :, :, :]
        self.nums[class_id-1].append(torch.sum(stack).item())

        return stack
    
    def load_all_stacks(
            self
    ) -> None:
        self.image_stacks = []
        self.liver_label_stacks = []
        self.tumour_label_stacks = []

        for stack_num in self.data_dict:
            self.image_stacks.append(
                self.load_image_stack(self.data_dict[stack_num]["image"])
                )
            self.liver_label_stacks.append(
                self.load_label_stack(
                    self.data_dict[stack_num]["mask"], 
                    class_id=1
                    )
                )
            self.tumour_label_stacks.append(
                self.load_label_stack(
                    self.data_dict[stack_num]["mask"], 
                    class_id=2
                    )
            )

    def build_flattened_dataset(
            self, 
    ) -> None:
        self.data_list = []
        dim_dict = {
            "axial": self.image_stacks[0].shape[1],
            "coronal": self.image_stacks[0].shape[2],
            "sagittal": self.image_stacks[0].shape[3]
        }
        for axis in self.axes:
            for i in range(len(self.image_stacks)):
                image_stack = self.image_stacks[i]
                liver_label_stack = self.liver_label_stacks[i]
                tumour_label_stack = self.tumour_label_stacks[i]

                for j in range(dim_dict[axis]):
                    if axis == "axial":
                        liver_label_selection = liver_label_stack[:, j, :, :]
                        tumour_label_selection = tumour_label_stack[:, j, :, :]

                        if torch.max(tumour_label_selection).item() > 0.:
                            image = image_stack[:, j, :, :]
                            liver_label = liver_label_selection
                            tumour_label = tumour_label_selection
                            self.data_list.append({"image": image, "liver_label": liver_label, "tumour_label": tumour_label,"axis": axis})
    
                        
                    elif axis == "coronal":
                        liver_label_selection = liver_label_stack[:, :, j, :]
                        tumour_label_selection = tumour_label_stack[:, :, j, :]

                        if torch.max(tumour_label_selection).item() > 0.: 
                            image_selection = image_stack[:, :, j, :]
                            image = torch.flip(image_selection, dims=[-2, -1])
                            liver_label = torch.flip(liver_label_selection, dims=[-2, -1])
                            tumour_label = torch.flip(tumour_label_selection, dims=[-2, -1])
                            self.data_list.append({"image": image, "liver_label": liver_label, "tumour_label": tumour_label,"axis": axis})

                    elif axis == "sagittal":
                        liver_label_selection = liver_label_stack[:, :, :, j]
                        tumour_label_selection = tumour_label_stack[:, :, :, j]

                        if torch.max(tumour_label_selection).item() > 0.: 
                            image_selection = image_stack[:, :, :, j]
                            image = torch.flip(image_selection, dims=[-2])
                            liver_label = torch.flip(liver_label_selection, dims=[-2])
                            tumour_label = torch.flip(tumour_label_selection, dims=[-2])
                            self.data_list.append({"image": image, "liver_label": liver_label, "tumour_label": tumour_label, "axis": axis})
           
        del self.image_stacks
        del self.liver_label_stacks
        del self.tumour_label_stacks

    def __getitem__(
            self, 
            index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        if index >= len(self.data_list):
            index = np.random.randint(0, len(self.data_list) - 1)
        
        data_dict = self.data_list[index]
        image, liver_label, tumour_label, axis = data_dict["image"], data_dict["liver_label"], data_dict["tumour_label"],data_dict["axis"]

        # if self.normalize_slices:
            # image = image / torch.max(image)
            

        image = self.image_resizer(torch.repeat_interleave(image, 3, dim=0))
        image = (image - self.mean) / self.std
        liver_label = liver_label.float()
        tumour_label = tumour_label.float()
        
        if self.training:
            return image, liver_label, axis
        else:
            return image, liver_label, tumour_label, axis
    
    def __len__(
        self
    ) -> int: 
        return len(self.data_list)

# if __name__ == "__main__":
#     dataloader = build_dataloader(axes=["axial"])
#     for i, (image, liver_label, tumour_label, axis) in enumerate(dataloader):
#         num_tumour = torch.sum(tumour_label)
#         num_liver = torch.sum(liver_label)
#         print((num_tumour/num_liver).item() * 100)            
