import torch
import sys
from typing import Tuple, Callable, List

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

from seg_models.sam_swag.segment_anything import sam_model_registry
from seg_models.sam_swag.swag_utils.swag import SWAG
from my_scripts.sam_final_scripts.sam_super import Sam_Super
from my_utils.final_utils.schedulers import SwagScheduler
from my_utils.final_utils.evaluator_2 import Evaluator
from my_utils.final_utils.evaluate_tumours_2 import TumourEvaluator
from my_utils.ensemble_predictive_utils import get_dice

PRETRAINED_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/seg_models/sam/checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
DEVICE = "cuda:0"
IMG_SIZE = 512

class Sam_Swag(Sam_Super):
    def __init__(
            self,   
            seed: int,
            max_num_models: int,
            swag_start: int, 
            swag_lr: float, 
            swag_freq: int, 
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

        self.model = sam_model_registry[MODEL_TYPE](checkpoint=PRETRAINED_PATH).to(self.device)
        self.main_params = list(self.model.image_encoder.parameters()) + list(self.model.mask_decoder.parameters()) + list(self.model.prompt_encoder.parameters())
        self.swag_model = SWAG(
            base=self.model.mask_decoder.to("cpu"), 
            max_num_models=max_num_models
            )
        self.model.mask_decoder.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.main_params, 
            lr=lr_init, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=1e-4, 
            amsgrad=False
        )
        self.lr_schedule = SwagScheduler(
            base_scheduler=self.lr_schedule,
            swag_start=swag_start,
            swag_lr=swag_lr,
            swag_freq=swag_freq
        )
        self.sampling_decoder = self.swag_model.base
        self.num_samples = num_samples

        self.x_embs = []
        self.validating = False

    @torch.enable_grad()
    def train(
            self, 
            train_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]],
            val_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]]

    ) -> None:
        self.step = 1
        self.model.train()
        self.optimizer.zero_grad()
        self.stop = False
        best_model = self.model.state_dict
        self.do_swag = False
        self.swag_steps = 0
        dices = []
        epi_means = []
        ali_means = []

        while not self.stop:
            for (image, label, axis) in train_loader:
                label = label.to(self.device)
                if self.step >= self.num_steps:
                    self.stop = True
                    break

                loss = self.predict(
                    image=image, 
                    axis=axis, 
                    training=True,
                    label=label,
                    decoder=self.model.mask_decoder
                    )
                loss.backward()

                if self.step % self.accumulator == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.optimizer = self.lr_schedule(self.optimizer)

                if self.step % self.eval_every == 0 and not self.do_swag:
                    dice, epi_mean, ali_mean = self.validate(val_loader)
                    dices.append(dice)
                    epi_means.append(epi_mean)
                    ali_means.append(ali_mean)
                    if dice < self.best_dice:
                        print(f"Model got worse on validation; breaking at {self.step} steps with old dice {self.best_dice} -> new dice {dice}...")
                        self.do_swag = True
                        self.model.load_state_dict(best_model)
                    else:
                        self.best_dice = dice
                        best_model = self.model.state_dict()

                if self.do_swag or self.step > self.lr_schedule.swag_start:
                    if self.swag_steps > 200:
                        self.stop = True
                        break
                    self.swag_steps += 1
                    if self.step % self.lr_schedule.swag_freq  == 0:
                        self.swag_model.collect_model(self.model.mask_decoder.to("cpu"))
                        self.model.mask_decoder.to(self.device)

                self.step += 1
        return self.step - 200, dices, epi_means, ali_means

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
                multimask_output=self.multimask_output
            )
            hr_mask = self.model.postprocess_masks(lr_mask, (1024, 1024), (IMG_SIZE, IMG_SIZE)).float()
            prediction = torch.nn.functional.sigmoid(hr_mask)

        if training:
            self.x_embs.append(x_emb.detach().cpu().numpy())
            loss = torch.nn.functional.binary_cross_entropy(prediction, label)
            dice = get_dice(prediction, label.to(self.device).float())
            sys.stdout.write(f"\rStep = {str(self.step).zfill(4)}; loss = {loss.item()}; dice = {dice.item()}")
            return loss
        else:
            return prediction
        
    @torch.no_grad()
    def sample(
            self, 
            x_emb: torch.Tensor,
            box: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        if self.validating:
            prediction = self.predict(
                image=[x_emb, box], 
                axis=None, 
                training=False, 
                decoder=self.model.mask_decoder
                )
            return prediction
        else:
            self.swag_model.sample(scale=1., device=self.device)
            self.swag_model.base.to(self.device)
            prediction = self.predict(
                image=[x_emb, box], 
                axis=None, 
                training=False, 
                decoder=self.swag_model.base
                )
            self.swag_model.base.to("cpu")

        return prediction

    @torch.no_grad()
    def eval_normal(
        self, 
        loader, 
        evaluator
    ):
        self.eval_x_embs = []
        for i, (image, label, axis) in enumerate(loader):
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.autocast(device_type="cuda", enabled=True):
                x_emb = self.model.image_encoder(image)
                self.eval_x_embs.append(x_emb.detach().cpu().numpy())
                box = self.prompt_generator(axis[0], label).to(self.device)
                
            prediction_stack = torch.stack([self.sample(x_emb, box) for _ in range(self.num_samples)], dim=0)
            evaluator.add_slice(prediction_stack, label)
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

            with torch.autocast(device_type="cuda", enabled=True):
                x_emb = self.model.image_encoder(image)
                self.eval_x_embs.append(x_emb.detach().cpu().numpy())
                box = self.prompt_generator(axis[0]).to(self.device)
                
            prediction_stack = torch.stack([self.sample(x_emb, box) for _ in range(self.num_samples)], dim=0)
            evaluator.add_slice(prediction_stack, label, anomaly_label)
        evaluator.collate_dataset_metrics()
        return evaluator
    

    # @torch.no_grad()
    # def test_tumours(
    #     self, 
    #     test_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]],
    #     num_samples: int
    # ) -> None:
    #     self.test_tumours_evaluator = TumourEvaluator()
    #     for i, (image, organ_label, tumour_label, axis) in enumerate(test_loader):
    #         image = image.to(self.device)
    #         organ_label = organ_label.to(self.device)
    #         tumour_label = tumour_label.to(self.device)
            
    #         x_emb = self.model.image_encoder(image)
    #         box = self.prompt_generator(axis[0]).to(self.device)

    #         prediction_stack = []
    #         for k in range(num_samples):
    #             prediction_stack.append(self.sample(x_emb, box))
                
    #         prediction_stack = torch.stack(prediction_stack, dim=0)
    #         print(prediction_stack.shape)
    #         self.test_tumours_evaluator.add_slice(image, prediction_stack, organ_label, tumour_label)
    #     self.test_tumours_evaluator.collate_dataset_metrics()