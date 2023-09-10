import torch
import numpy as np

import torchvision
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, DataLoader
import torchio as tio

import SimpleITK as sitk
import json

from typing import List, Tuple, Dict
import sys

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

def build_dataloaders(
        img_size: int = 512,
        indices: int = [1],
        training: bool = True,
        val_size: float = 480,
        normalize_slices: bool = False,
        axes: str = "all",
        class_selection : List[int] = [6],
        window_level: List[int] = [0, 0],
        noise_sigma: float = 0.,
        add_boxes: bool = False,
        angle: float = 0,
        data_dict_path: str = "/vol/biomedic3/bglocker/msc2023/cw1422/code/my_dataloaders/abdo/data_info.json",
        mean: float = None,
        std: float = None,
        batch_size: int = None,
        channels: int = 3,
        target_size: int = 1024
) -> Tuple[List, List, List]:
    
    with open(data_dict_path, 'r') as f:
        data_dict = json.load(f)
    
    data_dict = {key: data_dict[key] for key in data_dict if int(key) in indices}
    dataset = AbdoDataset(
        data_dict=data_dict, 
        normalize_slices=normalize_slices,
        class_selection=class_selection,
        axes=axes,
        img_size=img_size,
        window_level=window_level,
        noise_sigma=noise_sigma,
        add_boxes=add_boxes,
        angle=angle,
        mean=mean,
        std=std,
        target_size=target_size,
        channels=channels
    )
    
    if training:
        train_size = dataset.__len__() - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        

        dataloaders = {
            "train": DataLoader(
                train_set, 
                batch_size=1 if batch_size==None else batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                drop_last=False
            ),
            "val": DataLoader(
                val_set, 
                batch_size=1 if batch_size==None else batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False
            )
        }
        organ_dict, mean, std = dataset.fetch_organ_dict()
        return dataloaders, organ_dict, mean, std
    else:
        discard_size = dataset.__len__() - val_size
        dataset, discarded_set = torch.utils.data.random_split(dataset, [val_size, discard_size])
        dataloader = DataLoader(
            dataset, 
            batch_size=1 if batch_size==None else batch_size, 
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
            normalize_slices: bool = False,
            img_size: int = 512,
            class_selection: List[int] = [6],
            axes: str = "all",
            window_level: List[int] = [0, 0],
            noise_sigma: float = 0.,
            add_boxes: bool = False,
            angle: float = 0.,
            mean: float = None,
            std: float = None,
            target_size: int = None,
            channels: int = 3
    ) -> None:
        
        self.data_dict = data_dict
        self.normalize_slices = normalize_slices
        self.img_size = img_size
        self.class_selection = class_selection
        self.noise_sigma = noise_sigma
        self.angle = angle
        self.add_boxes = add_boxes
        self.image_resizer = transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.channels = channels
        self.window_level = window_level
        self.mean = 0
        self.std = 0
        
        self.train_mean = mean
        self.train_std = std

        if axes == "all":
            self.axes = ["axial", "coronal","sagittal"]
        else:
            self.axes = axes
        
        self.load_all_stacks()
        self.build_flattened_dataset()
        self.fill_second_organ_dict()

    def fetch_organ_dict(self):
        return self.second_organ_dict, self.mean, self.std

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
        
        # normaliser = tio.RescaleIntensity(
        #     out_min_max=(0, 255), 
        #     in_min_max=(torch.min(stack).item(), torch.max(stack).item())
        #     )
        
        # stack = normaliser(stack[None, :, : :])
        stack = stack - stack.min()
        # print(stack.float().mean(), stack.float().min(), stack.float().max())
        self.mean += stack.float().mean() / len(self.data_dict)
        self.std += stack.float().std() / len(self.data_dict)
        
        stack = stack[None, :, :, :]
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
        return stack
        
    def load_all_stacks(
            self
    ) -> None:
        self.image_stacks = []
        self.label_stacks = []

        for stack_num in self.data_dict:
            self.image_stacks.append(
                self.load_image_stack(self.data_dict[stack_num]["image"])
                )
            self.label_stacks.append(
                self.load_label_stack(
                    self.data_dict[stack_num]["label"], 
                    self.class_selection[0]
                    )
                )
            
    def init_first_organ_dict(self):
        variables = ["centroid", "tlx", "tly", "brx", "bry"]
        self.first_organ_dict = {axis: {variable: [] for variable in variables} for axis in self.axes}
    
    def fill_first_organ_dict(self, axis, label):
        locs = torch.where(label != 0)[1:]
        
        self.first_organ_dict[axis]["centroid"].append([torch.mean(locs[1].float()).numpy(), torch.mean(locs[0].float()).numpy()])
        self.first_organ_dict[axis]["tlx"].append(torch.min(locs[1].float()).numpy())
        self.first_organ_dict[axis]["tly"].append(torch.min(locs[0].float()).numpy())
        self.first_organ_dict[axis]["brx"].append(torch.max(locs[1].float()).numpy())
        self.first_organ_dict[axis]["bry"].append(torch.max(locs[0].float()).numpy())

    def init_second_organ_dict(self):
        variables = ["centroid_mean", "centroid_var", "tlx_mean", "tlx_var", "tly_mean", "tly_var", "brx_mean", "brx_var", "bry_mean", "bry_var"]
        self.second_organ_dict = {axis: {variable: None for variable in variables} for axis in self.axes}

    def fill_second_organ_dict(self):
        self.init_second_organ_dict()
        for axis in self.axes:
            centroid_arr = np.array(self.first_organ_dict[axis]["centroid"])
            mean_mean = np.array([np.mean(centroid_arr[:, 0]), np.mean(centroid_arr[:, 1])])
            cov = np.cov(centroid_arr[:, 0], centroid_arr[:, 1])

            self.second_organ_dict[axis]["centroid_mean"] = mean_mean
            self.second_organ_dict[axis]["centroid_var"] = cov

            self.second_organ_dict[axis]["tlx_mean"] = np.mean(np.array(self.first_organ_dict[axis]["tlx"]))
            self.second_organ_dict[axis]["tlx_var"] = np.std(np.array(self.first_organ_dict[axis]["tlx"]))
            self.second_organ_dict[axis]["tly_mean"] = np.mean(np.array(self.first_organ_dict[axis]["tly"]))
            self.second_organ_dict[axis]["tly_var"] = np.std(np.array(self.first_organ_dict[axis]["tly"]))

            self.second_organ_dict[axis]["brx_mean"] = np.mean(np.array(self.first_organ_dict[axis]["brx"]))
            self.second_organ_dict[axis]["brx_var"] = np.std(np.array(self.first_organ_dict[axis]["brx"]))
            self.second_organ_dict[axis]["bry_mean"] = np.mean(np.array(self.first_organ_dict[axis]["bry"]))
            self.second_organ_dict[axis]["bry_var"] = np.std(np.array(self.first_organ_dict[axis]["bry"]))

    def build_flattened_dataset(
            self, 
    ) -> None:
        self.data_list = []
        dim_dict = {
            "axial": self.image_stacks[0].shape[1],
            "coronal": self.image_stacks[0].shape[2],
            "sagittal": self.image_stacks[0].shape[3]
        }
        self.init_first_organ_dict()
        for axis in self.axes:
            for i in range(len(self.image_stacks)):
                image_stack = self.image_stacks[i]
                label_stack = self.label_stacks[i]
                for j in range(dim_dict[axis]):
                    if axis == "axial":
                        label_selection = label_stack[:, j, :, :]
                        if torch.max(label_selection).item() > 0.:
                            image = image_stack[:, j, :, :]
                            label = label_selection
                            self.data_list.append({"image": image, "label": label, "axis": axis})
                            self.fill_first_organ_dict(axis, label)
                        
                    elif axis == "coronal":
                        label_selection = label_stack[:, :, j, :]
                        if torch.max(label_selection).item() > 0.: 
                            image_selection = image_stack[:, :, j, :]
                            image = torch.flip(image_selection, dims=[-2, -1])
                            label = torch.flip(label_selection, dims=[-2, -1])
                            self.data_list.append({"image": image, "label": label, "axis": axis})
                            self.fill_first_organ_dict(axis, label)

                    elif axis == "sagittal":
                        label_selection = label_stack[:, :, :, j]
                        if torch.max(label_selection).item() > 0.: 
                            image_selection = image_stack[:, :, :, j]
                            image = torch.flip(image_selection, dims=[-2])
                            label = torch.flip(label_selection, dims=[-2])
                            self.data_list.append({"image": image, "label": label, "axis": axis})
                            self.fill_first_organ_dict(axis, label)
           
        del self.image_stacks
        del self.label_stacks
        
        if self.train_mean != None:
            self.mean = self.train_mean
            self.std = self.train_std

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        return vals, vecs
    
    def add_noise(self, image, label, sigma, angle):
        # image = transforms.ElasticTransform(alpha=float(abs(angle)*3))(image)
#         factor = 1 + (abs(angle)/200)
#         print(image.shape, label.shape)
#         zoomer = transformsv2.RandomZoomOut(
#             fill=image.min().item(), 
#             p=1., 
#             side_range=(factor, factor)
#         )
#         image = zoomer(image)
#         zoomer = transformsv2.RandomZoomOut(
#             fill=0.,
#             p=1., 
#             side_range=(factor, factor)
#         )
#         label = zoomer(label)
#         print(image.shape, label.shape)
        
#         angle = int(np.random.choice([angle, -angle], p=[0.5, 0.5]))
#         image = transforms.functional.rotate(image, angle, fill=image.min().item())
#         label = transforms.functional.rotate(label, angle, fill=image.min().item())
#         print(image.shape, label.shape)

        scale_factor = 1 - abs(angle) / 100
        translate_factor = 0.75*abs(angle) / 100
        affiner = transforms.RandomAffine(degrees=(angle, angle), translate=(translate_factor, translate_factor), scale=(scale_factor, scale_factor))
        
        state = torch.get_rng_state()
        image = affiner(image)
        torch.set_rng_state(state)
        label = affiner(label)
        
        image = image + sigma * torch.randn_like(image)
        return image, label
    
    def norm(self, image):
        image[image <= 0.] = 1e-3
        print("Before normed", image.max(), image.min())
        return 255 * (image - image.min()) / (image.max() - image.min())


    def patch_test(self, image, label, axis):
        sub_dict = self.second_organ_dict[axis]
        mean = np.copy(sub_dict["centroid_mean"])
        mean += np.random.uniform(low=-30, high=30, size=(2,))
        cov = np.copy(sub_dict["centroid_var"])
        cov = cov * np.random.uniform(0.01, 0.1)
        vals, vecs = self.eigsorted(cov)
        width, height = 2.5 * np.sqrt(vals)

        tlx = int(mean[0] - width / 2)
        tly =  int(mean[1] - height / 2)
        brx = int(mean[0] + width / 2)
        bry = int(mean[1] + height / 2)

        fill = torch.zeros((1, bry-tly, brx-tlx))
        fill = fill.normal_(mean=torch.mean(image), std=0.5*torch.std(image))
        anom_label = torch.zeros(label.shape)
        anom_label[:, tly:bry, tlx:brx] = 1
        image[:, tly:bry, tlx:brx] = fill
        label[:, tly:bry, tlx:brx] = 0

        return image, label, anom_label
        
    def __getitem__(
            self, 
            index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        if index >= len(self.data_list):
            index = np.random.randint(0, len(self.data_list) - 1)
        
        data_dict = self.data_list[index]
        image, label, axis = data_dict["image"], data_dict["label"], data_dict["axis"]

        
        if self.noise_sigma > 0:
            image, label = self.add_noise(image, label, self.noise_sigma, self.angle)
            if self.channels > 1:
                image = torch.repeat_interleave(image, self.channels, dim=0)
            else: 
                label = self.image_resizer(label)
            image = self.image_resizer(image)
            image = (image - self.mean) / self.std
            label = label.float()
            
            return image, label, axis
        elif self.add_boxes:
            
            image, label, anom_label = self.patch_test(image, label, axis)
            if self.channels > 1:
                image = torch.repeat_interleave(image, self.channels, dim=0)
            else:
                label = self.image_resizer(label)
            image = self.image_resizer(image)

            image = (image - self.mean) / self.std
            label = label.float()
            anom_label = anom_label.float()
            return image, label, anom_label, axis
        else:
            # if self.normalize_slices:
            #     image = self.norm(image)
            if self.channels > 1:
                image = torch.repeat_interleave(image, self.channels, dim=0)
            else:
                label = self.image_resizer(label)
            image = self.image_resizer(image)
            image = (image - self.mean) / self.std
            label = label.float()
            return image, label, axis
    
    def __len__(
        self
    ) -> int: 
        return len(self.data_list)

if __name__ == "__main__":
    dataloaders, organ_dict = build_dataloaders(transform_flag=True)
    for i, (image, label, anom_label, axis) in enumerate(dataloaders["train"]):
        print(anom_label.shape, torch.sum(anom_label))
