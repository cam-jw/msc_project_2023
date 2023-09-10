import numpy as np
import torch
import sys

sys.path.append("../")

from seg_models.sam_vanilla.segment_anything.utils.transforms import ResizeLongestSide

class Generator:
    def __init__(self, organ_dict):
        self.organ_dict = organ_dict
        self.prompt_resizer = ResizeLongestSide(1024)

    def __call__(
            self,
            axis,
            label=None  
    ):
        sub_dict = self.organ_dict[axis]
        # label = None
        
        if not label == None:
            locs = torch.where(label[0, 0, :, :] != 0)
            if not torch.numel(locs[0]) == 0:
                tlx = torch.min(locs[1].float()).item() - 25 #+ 1*np.random.normal(0, np.sqrt(sub_dict["tlx_var"]))
                tly = torch.min(locs[0].float()).item() - 25 #+ 1*np.random.normal(0, np.sqrt(sub_dict["tly_var"]))
                brx = torch.max(locs[1].float()).item() + 25#+ 1*np.random.normal(0, np.sqrt(sub_dict["brx_var"]))
                bry = torch.max(locs[0].float()).item() + 25 #+ 1*np.random.normal(0, np.sqrt(sub_dict["bry_var"]))
                box = np.array([tlx, tly, brx, bry])

                box = np.clip(box, 2, 510)
                box = self.prompt_resizer.apply_boxes(box, (512, 512))
                box = np.clip(box, 2, 1022)
                box = torch.from_numpy(box).float()

                return box

        tlx = sub_dict["tlx_mean"] + np.random.normal(0, 1*np.sqrt(sub_dict["tlx_var"]))
        tly = sub_dict["tly_mean"] + np.random.normal(0, 1*np.sqrt(sub_dict["tly_var"]))
        brx = sub_dict["brx_mean"] + np.random.normal(0, 1*np.sqrt(sub_dict["brx_var"]))
        bry = sub_dict["bry_mean"] + np.random.normal(0, 1*np.sqrt(sub_dict["bry_var"]))

        tlx -= 20
        tly -= 20
        brx += 20
        bry += 20

        box = np.array([tlx, tly, brx, bry])
        box = np.clip(box, 2, 510)
        box = self.prompt_resizer.apply_boxes(box, (512, 512))
        box = np.clip(box, 2, 1022)
        box = torch.from_numpy(box).float()

        return box

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        return vals, vecs