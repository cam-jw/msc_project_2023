import torch
import random
import numpy as np
import os
import sys
from typing import Tuple, Callable, List

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

from my_utils.ensemble_predictive_utils import get_dice
from my_utils.final_utils.schedulers import ExpScheduler
from my_utils.final_utils.evaluator_2 import Evaluator
from my_utils.final_utils.evaluate_tumours_2 import TumourEvaluator

PRETRAINED_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/seg_models/sam/checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
DEVICE = "cuda:0"
IMG_SIZE = 512

class Sam_Super:
    def __init__(
            self, 
            seed: int,
            lr_init: float,
            lr_decay: float,
            lr_freq: int,
            num_steps: int,
            accumulator: int, 
            prompt_generator: Callable[[str], torch.Tensor],
            # num_prompts: int,
            multimask_output: bool = False,
            device: str = DEVICE
    ) -> None:
        self.device = device
        self.lr_schedule = ExpScheduler(lr_init=lr_init, gamma=lr_decay, freq=lr_freq)
        self.num_steps = num_steps
        self.accumulator = accumulator
        self.prompt_generator = prompt_generator
        # self.num_prompts = num_prompts
        self.multimask_output = multimask_output
        self.set_seed(seed)
        self.eval_every = 10
        self.best_dice = 0

        self.epi_means = []
        self.epi_maxes = []

        self.ali_means = []
        self.ali_maxes = []

    def get_my_dice(
            self,
            prediction: torch.Tensor, 
            label: torch.Tensor
    ) -> float: 
        numerator = torch.sum(prediction * label)
        denominator = torch.sum(prediction**2) + torch.sum(label**2) + 1e-6
        return 1-((2*numerator) / denominator)

    def set_seed(
            self, 
            seed: int = 21
    ) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    def child_function(self, train_loader):
        pass

    def set_trainable_params(self):
        for n, value in self.model.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        for n, value in self.model.mask_decoder.transformer.named_parameters():
            if "MLP" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True
    def predict(
            self, 
            image: torch.Tensor, 
            axis: str,
            training: bool,
            decoder: Callable[[torch.Tensor], torch.Tensor],
            label=None
            ) -> torch.Tensor:
        image = image.to(self.device)
        if training:
            self.set_trainable_params()
            boxes = [self.prompt_generator(axis[0], label).to(self.device)]
            template = torch.zeros((1, 1, 1, IMG_SIZE, IMG_SIZE)).to(self.device)
        
        else:
            boxes = [self.prompt_generator(axis[0], label).to(self.device) for _ in range(self.num_prompts)]
            template = torch.zeros((self.num_prompts, 1, 1, IMG_SIZE, IMG_SIZE)).to(self.device)
        
        with torch.autocast(device_type="cuda", enabled=True):
            x_emb = self.model.image_encoder(
                image
            )
            for i, box in enumerate(boxes):
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
                template[i, :, :, :, :] = prediction
        
        prediction = torch.mean(template, dim=0)

        if training:
            loss = torch.nn.functional.binary_cross_entropy(prediction, label)
            dice = get_dice(prediction, label.to(self.device).float())
            sys.stdout.write(f"\rStep = {str(self.step).zfill(4)}; loss = {loss.item()}; dice = {dice.item()}")
            return loss
        else:
            return prediction

    @torch.enable_grad()
    def train(
            self, 
            train_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]],
            val_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]]
    ) -> None:
        self.step = 0
        self.model.train()
        self.optimizer.zero_grad()
        self.stop = False
        dices = []
        epi_means = []
        ali_means = []
        # best_model = self.model.state_dict
        while not self.stop:
            for (image, label, axis) in train_loader:
                # if self.step % self.eval_every == 0:
                #     dice, epi_mean, ali_mean = self.validate(val_loader)
                #     print("...val dice =", dice)
                #     dices.append(dice)
                #     epi_means.append(epi_mean)
                #     ali_means.append(ali_mean)
                    # if dice < self.best_dice:
                    #     print(f"Model got worse on validation; breaking at {self.step} steps with old dice {self.best_dice} -> new dice {dice}...")
                    #     # self.stop = True
                    #     self.model.load_state_dict(best_model)
                    #     # break
                    # else:
                    #     self.best_dice = dice
                    #     best_model = self.model.cpu().state_dict()
                    #     self.model.to(self.device)

                label = label.to(self.device).float()
                if self.step >= self.num_steps:
                    self.stop = True
                    break
                loss = self.predict(
                    image=image, 
                    axis=axis, 
                    label=label,
                    training=True,
                    decoder=self.model.mask_decoder
                    )
                loss.backward()

                if self.step % self.accumulator == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.optimizer = self.lr_schedule(self.optimizer)

                self.step += 1

                # if self.step % self.eval_every == 0:
                #     dice, epi_mean, ali_mean = self.validate(val_loader)
                #     print("...val dice =", dice)
                #     dices.append(dice)
                #     epi_means.append(epi_mean)
                #     ali_means.append(ali_mean)
                #     if dice < self.best_dice:
                #         print(f"Model got worse on validation; breaking at {self.step} steps with old dice {self.best_dice} -> new dice {dice}...")
                #         self.stop = True
                #         self.model.load_state_dict(best_model)
                #         break
                #     else:
                #         self.best_dice = dice
                #         best_model = self.model.cpu().state_dict()
                #         self.model.to(self.device)

            # self.step += 1

        self.child_function(train_loader)
        return self.step, dices, epi_means, ali_means

    @torch.no_grad()
    def sample(
            self, 
            image: torch.Tensor,
            axis: str
    ) -> torch.Tensor:
        self.model.eval()
        prediction = self.predict(
            image=image, 
            axis=axis, 
            training=False,
            decoder=self.sampling_decoder
            )
        return prediction
    
    @torch.no_grad()
    def validate(
        self, 
        loader
    ):
        self.validating = True
        evaluator = Evaluator()
        evaluator = self.eval_normal(
                loader,
                evaluator
            )
        dice = evaluator.organ_dict["predictive"][str(0.5)]["dice"]
        epi_mean = evaluator.organ_dict["epistemic_variance"]["mean"]
        ali_mean = evaluator.organ_dict["aleatoric_variance"]["mean"]
        epi_max = evaluator.organ_dict["epistemic_variance"]["max"]
        ali_max = evaluator.organ_dict["aleatoric_variance"]["max"]
        print()

        self.epi_means.append(epi_mean)
        self.epi_maxes.append(epi_max)

        self.ali_means.append(ali_mean)
        self.ali_maxes.append(ali_max)
        
        self.validating = False
        del evaluator
        self.model.train()
        torch.cuda.empty_cache()
        return dice, epi_mean, ali_mean
    
    # @torch.no_grad()
    # def validate(
    #     self, 
    #     val_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]], 
    #     num_samples: int
    #     ) -> None:
    #     self.val_evaluator = Evaluator()
    #     for i, (image, label, axis) in enumerate(val_loader):
    #         image = image.to(self.device)
    #         label = label.to(self.device).float()
    #         prediction_stack = []
    #         for j in range(num_samples):
    #             prediction = self.sample(image, axis)
    #             prediction_stack.append(prediction)
    #         prediction_stack = torch.stack(prediction_stack, dim=0)
    #         self.val_evaluator.add_slice(prediction_stack, label)
    #     self.val_evaluator.collate_dataset_metrics()

    # @torch.no_grad()
    # def test(
    #     self, 
    #     test_loaders: List[Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]]], 
    #     num_samples: int
    # ):
    #     self.test_evaluator = Evaluator()
    #     for i, loader in enumerate(test_loaders):
    #         for j, (image, label, axis) in enumerate(loader):
    #             sys.stdout.write(f"\r{i}/{len(test_loaders)}.....{j}/{len(loader)}.......")
    #             image = image.to(self.device)
    #             label = label.to(self.device).float()
    #             prediction_stack = []
    #             for k in range(num_samples):
    #                 prediction = self.sample(image, axis)
    #                 prediction_stack.append(prediction)
    #             prediction_stack = torch.stack(prediction_stack, dim=0)
    #             self.test_evaluator.add_slice(prediction_stack, label)
    #         if not i == len(test_loaders) - 1:
    #             self.test_evaluator.begin_new_stack()
    #     self.test_evaluator.collate_dataset_metrics()

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
    #         prediction_stack = []
    #         for k in range(num_samples):
    #             sample = self.sample(image, axis)
    #             print(sample.shape)
    #             prediction_stack.append(sample)
    #         prediction_stack = torch.stack(prediction_stack, dim=0)
    #         print(prediction_stack.shape)
    #         self.test_tumours_evaluator.add_slice(image, prediction_stack, organ_label, tumour_label)
    #     self.test_tumours_evaluator.collate_dataset_metrics()




