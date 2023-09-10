import torch
import sys
from typing import Tuple, Callable, List
import time

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

from seg_models.sam_single_pass.segment_anything import sam_model_registry
from my_scripts.sam_final_scripts.sam_super import Sam_Super
from my_utils.final_utils.evaluator_2 import Evaluator
from my_utils.final_utils.evaluate_tumours_2 import TumourEvaluator
from my_utils.ensemble_predictive_utils import get_dice

PRETRAINED_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/seg_models/sam/checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
DEVICE = "cuda:0"
IMG_SIZE = 512

class Sam_Single_Pass(Sam_Super):
    def __init__(
            self,   
            seed: int,
            num_masks_out: int,
            method: str,
            lr_init: float,
            lr_decay: float,
            lr_freq: int,
            num_steps: int,
            accumulator: int, 
            prompt_generator: Callable[[str], torch.Tensor],
            num_samples: int,
            device: str = DEVICE
    ) -> None:
        super().__init__(
            seed=seed,
            lr_init=lr_init,
            lr_decay=lr_decay,
            lr_freq=lr_freq,
            num_steps=num_steps,
            accumulator=accumulator, 
            prompt_generator=prompt_generator,
            # num_prompts=num_prompts,
            device=device
        )

        self.num_masks_out = num_masks_out
        self.model = sam_model_registry[MODEL_TYPE](checkpoint=PRETRAINED_PATH, num_masks_out=num_masks_out).to(self.device)
        self.main_params = list(self.model.image_encoder.parameters()) + list(self.model.mask_decoder.parameters()) + list(self.model.prompt_encoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.main_params, 
            lr=lr_init, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=1e-5, 
            amsgrad=False
        )
        self.sampling_decoder = self.model.mask_decoder.to(self.device)
        self.method = method
        self.num_samples = num_samples
        self.x_embs = []

    def predict(
            self, 
            image: torch.Tensor, 
            axis: str,
            training: bool,
            decoder: Callable[[torch.Tensor], torch.Tensor],
            label=None
            ) -> torch.Tensor:
        
        if training:
            image = image.to(self.device)
            self.set_trainable_params()
        
        with torch.autocast(device_type="cuda", enabled=True):
            if training:
                x_emb = self.model.image_encoder(
                    image
                )
                if label == None:
                    box = self.prompt_generator(axis[0]).to(self.device)
                else:
                    box = self.prompt_generator(axis[0], label).to(self.device)
            else:
                x_emb, box = image[0], image[1]

            sparse_prompt, dense_prompt = self.model.prompt_encoder(
                points=None, 
                boxes=box,
                masks=None
            )
            lr_mask, iou_pred = decoder.forward(
                image_embeddings=x_emb,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt,
                dense_prompt_embeddings=dense_prompt,
                multimask_output=self.multimask_output,
                method=self.method
            )
            hr_mask = self.model.postprocess_masks(lr_mask, (1024, 1024), (IMG_SIZE, IMG_SIZE)).float()
            prediction = torch.nn.functional.sigmoid(hr_mask)
        
        if training:
            self.x_embs.append(x_emb.detach().cpu().numpy())
            loss = torch.nn.functional.binary_cross_entropy(prediction, label)
            # loss = self.get_my_dice(prediction, label)
            dice = get_dice(prediction, label.to(self.device).float())
            sys.stdout.write(f"\rStep = {str(self.step).zfill(4)}; loss = {loss.item()}; dice = {dice.item()}")
            return loss
        else:
            self.eval_x_embs.append(x_emb.detach().cpu().numpy())
            return prediction
    
    @torch.no_grad()
    def sample(
            self, 
            image: torch.Tensor,
            label: torch.Tensor,
            axis: str
    ) -> torch.Tensor:
        self.model.eval()
        self.multimask_output = True
        image = image.to(self.device)
        with torch.autocast(device_type="cuda", enabled=True):
            x_emb = self.model.image_encoder(
                image
            )
            box = self.prompt_generator(axis[0], label).to(self.device)
            image = [x_emb, box]
            
        prediction = self.predict(
            image=image, 
            axis=axis, 
            training=False,
            decoder=self.model.mask_decoder
            )
        prediction = torch.swapaxes(prediction, 0, 1)[:, None, :, :, :]
        self.multimask_output = False
        return prediction, box

    @torch.no_grad()
    def eval_normal(
        self, 
        loader, 
        evaluator,
        flag=None
    ):
        self.eval_x_embs = []
        for i, (image, label, axis) in enumerate(loader):
            image = image.to(self.device)
            label = label.to(self.device)
            prediction_stack, box = self.sample(image, label, axis)
            evaluator.add_slice(prediction_stack, label, image=image, box=box, flag=flag)
        evaluator.collate_dataset_metrics()
        return evaluator
    
    @torch.no_grad()
    def eval_anomaly(
        self, 
        loader, 
        evaluator
    ):
        self.eval_x_embs = []
        for i, (image, label, anomaly_label, axis) in enumerate(loader):
            image = image.to(self.device)
            label = label.to(self.device)
            anomaly_label = anomaly_label.to(self.device)
            prediction_stack = self.sample(image, axis)
            evaluator.add_slice(prediction_stack, label, anomaly_label)
        evaluator.collate_dataset_metrics()
        return evaluator
            