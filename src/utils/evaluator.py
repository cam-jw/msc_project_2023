import torch
import numpy as np
from typing import List, Tuple, Dict
import sys
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import json
from torchio.transforms import RescaleIntensity
import tifffile 
import time
import torchvision

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

from .metrics import *

PREDICTION_THRESHOLDS = [np.round(0.1*i, 2) for i in range(11)]
UNCERTAIN_THRESHOLDS = [np.round(0.1*i, 2) for i in range(11)]
THRESHOLDS = [np.round(0.1*i, 2) for i in range(11)]
IMG_SHAPE = (1, 1, 512, 512)

class Evaluator:
    def __init__(
            self, 
            thresholds: List[float] = THRESHOLDS
    ) -> None:

        self.thresholds = thresholds

        self.buffer = {
            "organ_label": [[]],
            # "organ_label_down": [[]], 
            "mean": [[]], 
            # "mean_down": [[]],
            # "other_label": [[]],
            "predictive_entropy": [[]], 
            "predictive_variance": [[]],
            "expected_entropy": [[]],
            "aleatoric_variance": [[]],
            "mutual_information": [[]],
            "epistemic_variance": [[]]
        }
        self.predictive_metric_names = {
            "ece": get_ece, 
            "acc": get_accuracy, 
            "prec": get_precision, 
            "recall": get_recall, 
            "dice": get_dice, 
            "iou": get_iou, 
            "auroc": get_auroc
        }
        
        self.uncertain_metric_names = {
            "ece": get_ece, 
            "yarins": get_yarins, 
            "auroc": get_auroc
        }

        self.uncertain_candidate_names = [
            "predictive_entropy", 
            "predictive_variance", 
            "expected_entropy", 
            "aleatoric_variance",
            "mutual_information",
            "epistemic_variance"
            ]
        self.slices_added = 0

    def plot(
            self,
            image, 
            combined_dict,
            slice_index,
            box,
            flag
        ):

        
        fontsize = 20
        save_path = "/vol/biomedic3/bglocker/msc2023/cw1422/code/my_data/testing_2708/"
        cmap = "magma"
        plt.rcParams["font.family"] = "serif"

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 13))

        # plt.subplots_adjust(wspace=0.1, hspace=0.1) 
        if not image == None:
            box = box.detach().cpu().numpy()[0, :]

            tl = [box[0], box[1]]
            x_dist = box[2] - box[0]
            y_dist = box[3] - box[1]
            rect = patches.Rectangle((tl[0], tl[1]), x_dist, y_dist, facecolor='none', linewidth=1, edgecolor='r')
            
            axs[0, 0].imshow(image.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
            axs[0, 0].set_title("Image + Box", fontsize=fontsize, loc="left")
            axs[0, 0].add_patch(rect)
            axs[0, 0].axis("off")

        axs[0, 1].imshow(combined_dict["organ_label"].detach().cpu().numpy()[0, 0, :, :], cmap="gray")
        # rect = patches.Rectangle(int(tl[0]/2), int(tl[1]/2), int(x_dist/2), int(y_dist/2), facecolor='none', linewidth=1, edgecolor='r')
        # axs[0, 1].add_patch(rect)
        axs[0, 1].set_title("Label", fontsize=fontsize, loc="left")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(combined_dict["mean"].detach().cpu().numpy()[0, 0, :, :], cmap="gray")
        axs[0, 2].set_title("Probabilities", fontsize=fontsize, loc="left")
        axs[0, 2].axis("off")
        
        # axs[1, 0].imshow(combined_dict["predictive_entropy"].detach().cpu().numpy()[0, 0, :, :], cmap=cmap)
        # axs[1, 0].set_title("predictive entropy", fontsize=fontsize)
        # axs[1, 0].axis("off")

        # axs[1, 1].imshow(combined_dict["mutual_information"].detach().cpu().numpy()[0, 0, :, :], cmap=cmap)
        # axs[1, 1].set_title("mutual information", fontsize=fontsize)
        # axs[1, 1].axis("off")

        # axs[1, 2].imshow(combined_dict["expected_entropy"].detach().cpu().numpy()[0, 0, :, :], cmap=cmap)
        # axs[1, 2].set_title("expected entropy", fontsize=fontsize)
        # axs[1, 2].axis("off")

        kernel_size = 5
        stride = 1
        sigma = 4
        imm = combined_dict["predictive_variance"]
        # imm = torch.nn.functional.avg_pool2d(imm, kernel_size=kernel_size, stride=stride)
        imm = torchvision.transforms.functional.gaussian_blur(imm, kernel_size=kernel_size, sigma=sigma)
        imm = torch.nn.functional.avg_pool2d(imm, kernel_size=3)
        imm = imm.detach().cpu().numpy()[0, 0, :, :]
        im = axs[1, 0].imshow(imm, cmap=cmap)
        axs[1, 0].set_title("Predictive Variance", fontsize=fontsize, loc="left")
        axs[1, 0].axis("off")
        axins = axs[1, 0].inset_axes([0, -0.06, 1, 0.05])
        cbar = fig.colorbar(im, cax=axins, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks(np.linspace(0, np.max(imm), 2))
        cbar.ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        offset = cbar.ax.xaxis.get_offset_text()
        offset.set_fontsize(fontsize) 

        imm = combined_dict["epistemic_variance"]
        # imm = torch.nn.functional.avg_pool2d(imm, kernel_size=kernel_size, stride=stride)
        imm = torchvision.transforms.functional.gaussian_blur(imm, kernel_size=kernel_size, sigma=sigma)
        imm = torch.nn.functional.avg_pool2d(imm, kernel_size=3)
        imm = imm.detach().cpu().numpy()[0, 0, :, :]
        im = axs[1, 1].imshow(imm, cmap=cmap)
        axs[1, 1].set_title("Epistemic Variance", fontsize=fontsize, loc="left")
        axs[1, 1].axis("off")
        axins = axs[1, 1].inset_axes([0, -0.06, 1, 0.05])
        cbar = fig.colorbar(im, cax=axins, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks(np.linspace(0, np.max(imm), 2))
        cbar.ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        offset = cbar.ax.xaxis.get_offset_text()
        offset.set_fontsize(fontsize) 

        imm = combined_dict["aleatoric_variance"]
        # imm = torch.nn.functional.avg_pool2d(imm, kernel_size=kernel_size, stride=stride)
        imm = torchvision.transforms.functional.gaussian_blur(imm, kernel_size=kernel_size, sigma=sigma)
        imm = torch.nn.functional.avg_pool2d(imm, kernel_size=3)
        imm = imm.detach().cpu().numpy()[0, 0, :, :]
        im = axs[1, 2].imshow(imm, cmap=cmap)
        axs[1, 2].set_title("Aleatoric Variance", fontsize=fontsize, loc="left")
        axs[1, 2].axis("off")
        axins = axs[1, 2].inset_axes([0, -0.06, 1, 0.05])
        cbar = fig.colorbar(im, cax=axins, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks(np.linspace(0, np.max(imm), 2))
        cbar.ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        offset = cbar.ax.xaxis.get_offset_text()
        offset.set_fontsize(fontsize) 

        plt.tight_layout()
        save_path += f"{str(slice_index).zfill(4)}"
        if not flag == None:
            save_path += f"_{flag}"
        plt.savefig(save_path, dpi=1000, bbox_inches='tight') 
        plt.close()

    def add_slice(
            self,
            predictions: torch.Tensor,
            organ_label: torch.Tensor,
            kernel_size: int = 1,
            plot=True,
            box=None,
            image=None,
            flag=None,
            other_label=None
    ) -> None:
        self.slices_added += 1

        if predictions.shape[0] == 0:
            predictions = torch.repeat_interleave(predictions, repeats=2, dim=0)
            
        mean, uncertain_candidate_dict = get_all_uncertainties(predictions, kernel_size, mp=True)
        self.predictive_candidate_dict = {
            "organ_label": organ_label,
            # "organ_label_down": downsize(organ_label, kernel_size, mask=True),
            "mean": mean, 
            # "mean_down": downsize(mean, kernel_size, mask=False, mp=True)
        }
        if not other_label == None:
            self.predictive_candidate_dict["other_label"] = other_label

        combined_dict = {**self.predictive_candidate_dict, **uncertain_candidate_dict}
        for key in combined_dict:
            self.buffer[key][-1].append(combined_dict[key].cpu())

        # goods = [180, 200, 260, 280, 340, 380]
        if plot:
            # if self.slices_added in goods:
            if self.slices_added % 2 == 0:
                self.plot(image=image, combined_dict=combined_dict, slice_index=self.slices_added, box=box, flag=flag)

    def init_results_dict(
            self
    ) -> Dict[str, Dict]:
        organ_dict = {
            "predictive": {
                str(pred_threshold): {metric: 0. for metric in self.predictive_metric_names} \
                                    for pred_threshold in self.thresholds
                }, 
            "uncertain": {
                str(pred_threshold): {candidate: {metric: {str(sub_threshold): 0. for sub_threshold in self.thresholds} \
                                if metric == "yarins" else 0. \
                                    for metric in self.uncertain_metric_names} \
                                        for candidate in self.uncertain_candidate_names} \
                                            for pred_threshold in self.thresholds
                    }
        }
        return organ_dict

    def collate_dataset_metrics(
            self
    ) -> None:
        temp_lists = {key: [] for key in self.buffer}
        for stack_index in range(len(self.buffer["mean"])):
            for key in self.buffer:
                stack = torch.stack(self.buffer[key][stack_index], dim=0)
                temp_lists[key].append(stack)

        del self.buffer
        torch.cuda.empty_cache()
        self.full_dataset_dict = {key: 0 for key in temp_lists}
        # uncertain_candidates_normed_dict = {key: 0 for key in self.uncertain_candidate_names}
        for key in temp_lists:
            super_stack = torch.cat(tuple(sub_stack for sub_stack in temp_lists[key]), dim=0)
            self.full_dataset_dict[key] = super_stack #.cpu()
            # if key in self.uncertain_candidate_names:
            #     uncertain_candidates_normed_dict[key] = normalise(super_stack, constraint="None")
        del temp_lists
        torch.cuda.empty_cache()
        # organ_label, mean = tuple([self.full_dataset_dict[key] for key in self.predictive_candidate_dict])
        self.organ_dict = self.init_results_dict()
        organ_label = self.full_dataset_dict["organ_label"]

        # in_other = self.full_dataset_dict["epistemic_variance"] * self.full_dataset_dict["other_label"]
        # out_other = self.full_dataset_dict["epistemic_variance"] * (1-self.full_dataset_dict["other_label"])

        # in_other_pos = in_other[torch.where(in_other > 0)]
        # out_other_pos = out_other[torch.where(out_other > 0)]

        # in_other_mean = in_other_pos.mean().item()
        # out_other_mean = out_other_pos.mean().item()

        # in_other_var = in_other_pos.var().item()
        # out_other_var = out_other_pos.var().item()

        # print(in_other_mean, in_other_var, out_other_mean, out_other_var)

        # in_other_hist, in_other_bins = get_hist(in_other_pos, 1000)
        # out_other_hist, out_other_bins = get_hist(out_other_pos, 1000)

        # self.organ_dict["in_other"] = [in_other_hist, in_other_bins]
        # self.organ_dict["out_other"] = [out_other_hist, out_other_bins]
 
        for pred_treshold in self.thresholds:
            binary_pred = threshold(self.full_dataset_dict["mean"], pred_treshold)
            self.organ_dict["predictive"][str(pred_treshold)]["dice"] = get_dice(binary_pred, organ_label)
            
        self.organ_dict["mean_mean"] = torch.mean(self.full_dataset_dict["mean"]).item()
        for uncertain_name in self.uncertain_candidate_names:
            self.organ_dict[uncertain_name] = {
                "mean": torch.mean(self.full_dataset_dict[uncertain_name]).item(),
                "max": torch.max(self.full_dataset_dict[uncertain_name]).item(),
                "var": torch.var(self.full_dataset_dict[uncertain_name]).item()
                }
            
            self.organ_dict[f"{uncertain_name}_max"] = torch.max(self.full_dataset_dict[uncertain_name]).item()
            self.organ_dict[f"{uncertain_name}_var"] = torch.max(self.full_dataset_dict[uncertain_name]).item()

        print("doing metrics")
        for pred_threshold in self.thresholds:
            binary_pred = threshold(mean, pred_threshold)
            # binary_pred_down = threshold(mean_down, pred_threshold)
            ece_pseudolabel = (organ_label != binary_pred.float()).float()

            print("doing predictive")
            for predictive_metric in self.predictive_metric_names:
                function = self.predictive_metric_names[predictive_metric]
                if "ece" in predictive_metric or "auroc" in predictive_metric:
                    self.organ_dict["predictive"][str(pred_threshold)][predictive_metric] = function(prediction=mean, label=organ_label)
                else:
                    self.organ_dict["predictive"][str(pred_threshold)][predictive_metric] = function(prediction=binary_pred, label=organ_label)

            print("doing uncertain")
            for uncertain_candidate_name in self.uncertain_candidate_names:
                stack = uncertain_candidates_normed_dict[uncertain_candidate_name]
                self.organ_dict["uncertain"][str(pred_threshold)][uncertain_candidate_name]["ece"] = get_ece(stack, ece_pseudolabel)
                self.organ_dict["uncertain"][str(pred_threshold)][uncertain_candidate_name]["auroc"] = get_auroc(stack, ece_pseudolabel)

            for uncertain_threshold in self.thresholds:
                for uncertain_candidate_name in self.uncertain_candidate_names:
                    stack = uncertain_candidates_normed_dict[uncertain_candidate_name]
                    yarins = get_yarins(binary_pred, stack, organ_label, uncertain_threshold)
                    self.organ_dict["uncertain"][str(pred_threshold)][uncertain_candidate_name]["yarins"][str(uncertain_threshold)] = yarins

    def ndarray_to_list(self, data):
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                new_dict[key] = self.ndarray_to_list(value)
            return new_dict
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [self.ndarray_to_list(item) for item in data]
        else:
            return data

    def save_data(
            self,
            base_path: str
        ) -> None:

        base_path += "/"

        dicts_to_save = [self.organ_dict]
        dicts_to_save = list(map(lambda x: self.ndarray_to_list(x), dicts_to_save))
        dict_names = ["organ"]
    
        for i, dict_to_save in enumerate(dicts_to_save):
            save_path = base_path + f"_{dict_names[i]}.json"
            with open(save_path, "w") as f:
                json.dump(dict_to_save, f, indent=4)

        for key in self.full_dataset_dict:
            if "epistemic" in key or "aleatoric" in key or "mean" in key or "label" in key:
                with open(base_path + f"{key}.npy", "wb") as f:
                    np.save(
                            f,
                            self.full_dataset_dict[key][:, 0, 0, :, :].cpu().numpy()
                    )
        del self.full_dataset_dict
        torch.cuda.empty_cache()
