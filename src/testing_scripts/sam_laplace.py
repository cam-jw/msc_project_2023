import torch
from torchvision import transforms
import sys
from typing import Tuple, Callable, List
from laplace import Laplace
from laplace.utils import ModuleNameSubnetMask
from backpack import extend

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

from seg_models.sam_laplace.segment_anything import sam_model_registry
from seg_models.sam_laplace.laplace_utils.decoder import Decoder as LaplaceDecoder
from seg_models.sam_laplace.laplace_utils.encoder import Encoder as LaplaceEncoder
from seg_models.sam_laplace.laplace_utils.laplace_dataloader import build_laplace_dataloader
from my_scripts.sam_final_scripts.sam_super import Sam_Super
from my_utils.final_utils.evaluator_laplace import Evaluator

PRETRAINED_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/seg_models/sam/checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
DEVICE = "cuda:0"
IMG_SIZE = 512

class Sam_Laplace(Sam_Super):
    def __init__(
            self, 
            seed: int,
            parameter_fraction: int,
            lr_init: float,
            lr_decay: float,
            lr_freq: int,
            num_steps: int,
            accumulator: int, 
            prompt_generator: Callable[[str], torch.Tensor],
            num_prompts: int,
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
            num_prompts=num_prompts,
            device=device
        )

        self.parameter_fraction = parameter_fraction
        self.model = sam_model_registry[MODEL_TYPE](checkpoint=PRETRAINED_PATH).to(self.device)
        self.main_params = list(self.model.image_encoder.parameters()) + list(self.model.mask_decoder.parameters()) + list(self.model.prompt_encoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.main_params, 
            lr=lr_init, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=1e-4, 
            amsgrad=False
        )
        self.child_function = self.perform_laplace

    def perform_laplace(
            self, 
            train_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]],
            hypernetwork_index: int = 0,
            ):
        decoder = self.model.mask_decoder.output_hypernetworks_mlps[hypernetwork_index]
        self.laplace_decoder = extend(
            LaplaceDecoder(decoder),
            use_converter=True
        )
        self.laplace_encoder = LaplaceEncoder(
            image_encoder=self.model.image_encoder,
            prompt_encoder=self.model.prompt_encoder,
            original_decoder=self.model.mask_decoder,
            prompt_generator=self.prompt_generator,
            device=self.device
        )
        self.laplace_loader = build_laplace_dataloader(
            encoder=self.laplace_encoder,
            train_loader=train_loader,
            fitting=True,
            device=self.device
        )
        subnetwork_mask = ModuleNameSubnetMask(
            self.laplace_decoder, 
            module_names=['decoder_network.layers.0', 'decoder_network.layers.2', 'decoder_network.layers.4']
            )
        subnetwork_mask.select()
        subnetwork_indices = subnetwork_mask.indices.to("cpu")
        subnetwork_indices = torch.tensor([subnetwork_indices[i] for i in range(0, subnetwork_indices.shape[0], self.parameter_fraction)])
        self.laplace_object = Laplace(
            self.laplace_decoder,
            'classification',
            subset_of_weights="subnetwork",
            hessian_structure="full",
            subnetwork_indices=subnetwork_indices
        )
        self.laplace_object.fit(self.laplace_loader)

    def reshape_resize(
            self, 
            tensors: List[torch.Tensor], 
            target_size: int = 512
        ) -> Tuple[torch.Tensor]:
        outs = []
        for x in tensors:
            x = torch.reshape(x, (1, 1, 64, 64))
            x = torch.nan_to_num(x, 0.)
            # x = transforms.functional.resize(
            #     img=x, 
            #     size=(target_size, target_size),
            #     interpolation=transforms.InterpolationMode.BICUBIC,
            #     antialias=True
            #     )
            outs.append(x)
        return tuple(outs)

    @torch.no_grad()
    def sample(
            self, 
            image: torch.Tensor,
            axis: str
    ) -> torch.Tensor:
        encoded = self.laplace_encoder(image, axis[0])
        encoded = encoded[None, :, :, :, :].to(self.device)

        prediction_mean, prediction_covar, sample = self.laplace_object(
            encoded, 
            pred_type="glm",
            link_approx="mc",
            )
        
        prediction_var = torch.diagonal(prediction_covar[0, :, :])
        prediction_mean, prediction_var = self.reshape_resize([prediction_mean, prediction_var])
        return prediction_mean, prediction_var
    
    @torch.no_grad()
    def validate(
        self, 
        val_loader: Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]], 
        num_samples: int
        ) -> None:
        print("validating")
        self.val_evaluator = Evaluator()
        for i, (image, label, axis) in enumerate(val_loader):
            # if i>2: break
            image = image.to(self.device)
            label = label.to(self.device).float()
            mean_stack = []
            sys.stdout.write(f"\r{i}/{len(val_loader)}.......")
            for j in range(num_samples):
                mean, variance = self.sample(image, axis)
                mean_stack.append(mean)
            mean_stack = torch.stack(mean_stack, dim=0)
            self.val_evaluator.add_slice(mean_stack, mean, variance, label)
        self.val_evaluator.collate_dataset_metrics()

    @torch.no_grad()
    def test(
        self, 
        test_loaders: List[Callable[[None], Tuple[torch.Tensor, torch.Tensor, str]]], 
        num_samples: int
    ) -> None:  
        print("Testing")
        self.test_evaluator = Evaluator()
        print(len(test_loaders))
        for i, loader in enumerate(test_loaders):
            for j, (image, label, axis) in enumerate(loader):
                # if j > 3: break
                sys.stdout.write(f"\r{i}/{len(test_loaders)}.....{j}/{len(loader)}.......")
                image = image.to(self.device)
                label = label.to(self.device).float()
                mean_stack = []
                for k in range(num_samples):
                    mean, variance = self.sample(image, axis)
                    mean_stack.append(mean)
                mean_stack = torch.stack(mean_stack, dim=0)
                self.test_evaluator.add_slice(mean_stack, mean, variance, label)
            if not i == len(test_loaders) - 1:
                self.test_evaluator.begin_new_stack()
        self.test_evaluator.collate_dataset_metrics()