
import sys
import os
import json
import shutil
import numpy as np
import torch
import random
import time

BASE_PATH = "/vol/biomedic3/bglocker/msc2023/cw1422/code/"
sys.path.append(BASE_PATH)

from ..dataloaders.abdo_dataloader import build_dataloaders as build_dataloaders_abdo
from ..dataloaders.abdo_tumour_dataloader import build_dataloader as build_dataloader_tumours
from ..dataloaders.prompt_generator import Generator as PromptGenerator

from .sam_multi_decoder import Sam_Single_Pass
from .sam_multi_prompt import Sam_Multi_Prompt
from .sam_mc_dropout import Sam_MC
from .sam_single_swag import Sam_Swag
from .sam_single_bbb import Sam_Bayes
from .sam_multi_bbb import Sam_Multi_Bayes
from .sam_multi_swag import Sam_Multi_Swag

from ..utils.evaluator import Evaluator

DEVICE = "cuda:0"

class Tester:
    def __init__(self, device=DEVICE):
        self.organ_dict = {
            "liver": 6
        }

        self.models = {
        # Single Decoders
            "zero_shot": self.init_base_sam,
            "vanilla": self.init_base_sam,
            "poly_prompt": self.init_multi_prompt,
            "monte_carlo": self.init_monte_carlo,
            "swag": self.init_swag,
            "bayes": self.init_bayes,
        #     # Multi Decoders - Mean Output
            "multi_zero_shot": self.init_multi_decoder,
            "multi_decoder_mean": self.init_multi_decoder,
            "multi_swag_mean": self.init_multi_swag,
            "multi_bayes_mean": self.init_multi_bayes,
            "super_multi_bayes_mean": self.init_super_bayes,
        #     # Multi Decoders - Mean Output
            "multi_decoder_sample": self.init_multi_decoder,
            "multi_swag_sample": self.init_multi_swag,
            "multi_bayes_sample": self.init_multi_bayes,
            "super_multi_bayes_sample": self.init_super_bayes
        }

        self.num_samples = 3
        self.seeds = [101, 203, 305]
        self.device = device
        self.base_window_level = [400, 50]
        
        # self.base_path = "/vol/biomedic3/bglocker/msc2023/cw1422/code/results/ood_testing_2708_final/"
        self.base_path = "/vol/biomedic3/bglocker/msc2023/cw1422/code/results/trial/"
        # self.base_path = "/vol/biomedic3/bglocker/msc2023/cw1422/code/results/ood_testing_3108_final/"

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

    def get_training_generator(self, organ, indices, val_size):
        dataloaders, organ_dict, mean, std = build_dataloaders_abdo(
            indices=indices,
            training=True,
            axes=["axial"],
            class_selection=[organ],
            val_size=val_size,
            window_level=self.base_window_level,
            normalize_slices=True
        )
        prompt_generator = PromptGenerator(organ_dict)
        train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        return train_loader, prompt_generator, mean, std
        
    def get_validation(self, organ_int, test_indices):
        return self.val_loader
    
    def get_development(self, organ_int, indices, val_size, mean, std, sigma=0., angle=0.):
        test_loader = build_dataloaders_abdo(
            indices=indices,
            training=False,
            axes=["axial"],
            class_selection=[organ_int],
            window_level=self.base_window_level,
            normalize_slices=True,
            val_size=val_size,
            mean=mean, 
            std=std,
            noise_sigma=sigma,
            angle=angle
        )
        return test_loader
        
    def get_test(self, organ, indices, window_level, sigma, angle, val_size, mean, std):
        test_loader = build_dataloaders_abdo(
            indices=indices,
            training=False,
            axes=["axial"],
            class_selection=[organ],
            window_level=window_level,
            normalize_slices=True,
            noise_sigma=sigma, 
            angle=angle,
            val_size=val_size,
            mean=mean,
            std=std
        )
        return test_loader
    
    def init_base_sam(
        self, 
        seed,
        num_samples,
        prompt_generator, 
        num_steps
    ):
        model_wrapper = Sam_Multi_Prompt(
            seed=seed,
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=1,
            device=DEVICE
        )
        return model_wrapper
    
    def init_multi_prompt(
        self, 
        seed,
        num_samples,
        prompt_generator, 
        num_steps
    ):
        model_wrapper = Sam_Multi_Prompt(
            seed=seed,
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            device=DEVICE
        )
        return model_wrapper
    
    def init_multi_decoder(
            self,
            seed, 
            num_samples, 
            prompt_generator, 
            num_steps,
            method="sample"
            ):
        model_wrapper = Sam_Single_Pass(
            seed=seed,
            num_masks_out=num_samples, 
            method=method,
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            device=DEVICE
            )
        return model_wrapper
    
    def init_monte_carlo(
            self, 
            seed,
            num_samples,
            prompt_generator,
            num_steps
    ):
        model_wrapper = Sam_MC(
            seed=seed,
            enc_dropout=0.0001, 
            dec_dropout=0.2, 
            lr_init=8e-5,
            lr_decay=0.999,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            device=DEVICE
            )
        return model_wrapper
    
    def init_swag(
            self, 
            seed, 
            num_samples, 
            prompt_generator, 
            num_steps
    ):
        model_wrapper = Sam_Swag(
            seed=seed,
            max_num_models=20, 
            swag_start=num_steps, 
            swag_lr=1e-2,
            swag_freq=10,
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps+200,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            device=DEVICE
            )
        return model_wrapper
    
    def init_multi_swag(
            self, 
            seed, 
            num_samples, 
            prompt_generator, 
            num_steps,
            method
    ):
        model_wrapper = Sam_Multi_Swag(
            seed=seed,
            max_num_models=20, 
            swag_start=num_steps, 
            swag_lr=1e-2,
            swag_freq=10,
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps+200,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            num_masks_out=num_samples,
            method=method,
            device=DEVICE
            )
        return model_wrapper

    def init_bayes(
        self, 
        seed, 
        num_samples, 
        prompt_generator, 
        num_steps
    ):
        beta = 1
        model_wrapper = Sam_Bayes(
            seed=seed,
            enc_bayes_freq="none", 
            dec_bayes=True, 
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            beta=beta,
            device=DEVICE
            )
        return model_wrapper
    
    def init_multi_bayes(
            self, 
            seed, 
            num_samples, 
            prompt_generator, 
            num_steps,
            method
    ):
        beta = 1
        model_wrapper = Sam_Multi_Bayes(
            seed=seed,
            enc_bayes_freq="none", 
            dec_bayes=True, 
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            beta=beta,
            num_masks_out=num_samples,
            method=method,
            device=DEVICE
            )
        return model_wrapper
    
    def init_super_bayes(
            self, 
            seed, 
            num_samples, 
            prompt_generator, 
            num_steps,
            method
    ):
        beta = 1
        model_wrapper = Sam_Multi_Bayes(
            seed=seed,
            enc_bayes_freq="last", 
            dec_bayes=True, 
            lr_init=8e-5,
            lr_decay=0.99,
            lr_freq=100,
            num_steps=num_steps,
            accumulator=1,
            prompt_generator=prompt_generator,
            num_samples=num_samples,
            beta=beta,
            num_masks_out=num_samples,
            method=method,
            device=DEVICE
            )
        return model_wrapper
    
    def main(self):
        iteration_dict_0 = {
            "0": {
                "train": [i for i in range(8, 24)],
                "dev": [3, 4, 5, 6, 7],
                "test": [1, 2, 4, 25, 26, 27, 28, 29]
            } ,
            "1": {
                "train": [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 21, 23, 24],
                "dev": [7, 8, 9, 10, 11],
                "test": [12, 13, 14, 15, 16, 17, 25, 26]
            }
        }

        iteration_dict_1 = {
            "0": {
                "train": [i for i in range(0, 16)],
                "dev": [16, 17, 18, 19, 20],
                "test": [20, 21, 22, 23, 24, 25, 26, 27]
            } ,
            "1": {
                "train": [i for i in range(16, 29)],
                "dev": [1, 2, 3, 4, 5],
                "test": [6, 7, 8, 9, 10, 11, 12, 13]
            }
        }

        num_samples = 3
        self.organ_name = "liver"
        organ_int = self.organ_dict[self.organ_name]
        val_size = 100
        
        self.window_levels = [[400, 50], [325, 50], [250, 50], [175, 50], [100, 50], [50, 50]]
        self.sigmas = [0., 20., 40., 60., 80., 100.]
        self.angles = [0, 15, 30, 45, 60]
        self.seeds = [101]

        out_dict = {model_name: {} for model_name in self.models}
        for seed in self.seeds:
            for model_name in self.models:
                if seed == 101:
                    iteration_dict = iteration_dict_1
                else:
                    iteration_dict = iteration_dict_0
                for iteration in iteration_dict:
                    print(model_name, iteration)
                    train_indices = iteration_dict[iteration]["train"]
                    dev_indices = iteration_dict[iteration]["dev"]
                    test_indices = iteration_dict[iteration]["test"]
                    self.set_seed(seed)
                    print("getting train loader")
                    train_loader, generator, train_mean, train_std = self.get_training_generator(organ_int, train_indices, val_size=val_size)
                    print("getting dev loader")
                    dev_loader = self.get_development(organ_int=organ_int, indices=dev_indices, val_size=val_size, mean=train_mean, std=train_std)

                    if "multi" in model_name and not "zero" in model_name:
                        if "mean" in model_name:
                            method = "mean"
                        else:
                            method = "sample"
                        model_wrapper = self.models[model_name](
                            seed=seed, 
                            num_samples=num_samples, 
                            prompt_generator=generator, 
                            num_steps=2000,
                            method=method
                            )
                    elif "zero" in model_name:
                        model_wrapper = self.models[model_name](
                            seed=seed, 
                            num_samples=num_samples, 
                            prompt_generator=generator, 
                            num_steps=2
                            )
                    else:
                        model_wrapper = self.models[model_name](
                            seed=seed, 
                            num_samples=num_samples, 
                            prompt_generator=generator, 
                            num_steps=2
                            )

                    start_time = time.time()
                    print("training")
                    num_steps, dices, epi_means, ali_means = model_wrapper.train(train_loader, dev_loader)
                    
                    print(f"Done {model_name}")

                    self.set_seed(seed)
                    anomaly_dl = build_dataloader_tumours(
                        indices = [6],
                        training = True,
                        normalize_slices=True,
                        axes=["axial"],
                        window_level = [400, 50],
                        mean = train_mean,
                        std = train_std
                    )
                    evaluator = Evaluator()
                    evaluator = model_wrapper.eval_normal(anomaly_dl, evaluator, flag=f"anomaly")

                    for k in range(len(self.angles)+1):
                        self.set_seed(seed)
                        print(k)
                        if k == 0:
                            test_loader = self.get_validation(organ_int=organ_int, test_indices=test_indices)
                        else:
                            test_loader = self.get_test(
                                organ_int, 
                                test_indices, 
                                self.base_window_level,
                                1,
                                self.angles[k-1],
                                val_size=val_size,
                                mean=train_mean,
                                std=train_std
                            )

                        evaluator = Evaluator()
                        evaluator = model_wrapper.eval_normal(test_loader, evaluator, flag=f"space_{k}")
                        evaluator.organ_dict["num_steps"] = num_steps
                        evaluator.organ_dict["dices"] = dices
                        evaluator.organ_dict["epi_means"] = epi_means
                        evaluator.organ_dict["ali_means"] = ali_means

                        final_dirs = os.path.join(
                            self.base_path,
                            "seed_" + str(seed), 
                            self.organ_name, 
                            model_name, 
                            "train_size_" + str(len(train_indices)),
                            "num_samples_" + str(num_samples), 
                            "iteration_" + str(iteration), 
                            "space_deg_factor_" + str(k)
                        )

                        os.makedirs(final_dirs)
                        # print("saving data")
                        evaluator.save_data(final_dirs)
                        # print("finished saving")

                        # print("saving x embeddings")
                        # x_emb_dir = final_dirs + "/x_embs.npy"
                        # with open(x_emb_dir, "wb") as f:
                        #     np.save(f, np.array(model_wrapper.eval_x_embs))

                        # print("deleting objects")
                        del evaluator
                        del test_loader
                        # print("clearing cache")
                        
                        torch.cuda.empty_cache()

                    for k in range(1, len(self.sigmas)):
                        self.set_seed(seed)
                        print(k)
                        print(self.sigmas[k])
                        test_loader = self.get_test(
                            organ_int, 
                            test_indices, 
                            self.window_levels[k], 
                            self.sigmas[k], 
                            0.01,
                            val_size=val_size,
                            mean=train_mean,
                            std=train_std
                        )

                        evaluator = Evaluator()
                        evaluator = model_wrapper.eval_normal(test_loader, evaluator, flag=f"intensity_{str(k+1)}")
                        evaluator.organ_dict["num_steps"] = num_steps
                        evaluator.organ_dict["dices"] = dices
                        evaluator.organ_dict["epi_means"] = epi_means
                        evaluator.organ_dict["ali_means"] = ali_means

                        final_dirs = os.path.join(
                            self.base_path,
                            "seed_" + str(seed), 
                            self.organ_name, 
                            model_name, 
                            "train_size_" + str(len(train_indices)),
                            "num_samples_" + str(num_samples), 
                            "iteration_" + str(iteration), 
                            "intensity_deg_factor_" + str(k+1)
                        )

                        os.makedirs(final_dirs)
                        evaluator.save_data(final_dirs)

                        x_emb_dir = final_dirs + "/x_embs.npy"
                        with open(x_emb_dir, "wb") as f:
                            np.save(f, np.array(model_wrapper.eval_x_embs))

                        del evaluator
                        del test_loader
                        
                        torch.cuda.empty_cache()

                    self.set_seed(seed)
                    anomaly_dl = build_dataloaders_abdo(
                        indices = test_indices,
                        training = False,
                        val_size = val_size,
                        normalize_slices=True,
                        axes=["axial"],
                        class_selection=[6],
                        window_level = [400, 50],
                        noise_sigma = 0,
                        add_boxes = True,
                        angle = 0,
                        mean = train_mean,
                        std = train_std
                    )
                    evaluator = Evaluator()
                    evaluator = model_wrapper.eval_normal(anomaly_dl, evaluator, flag=f"anomaly")

                    final_dirs = os.path.join(
                            self.base_path,
                            "seed_" + str(seed), 
                            self.organ_name, 
                            model_name, 
                            "train_size_" + str(len(train_indices)),
                            "num_samples_" + str(num_samples), 
                            "iteration_" + str(iteration), 
                            "anomaly"
                        )

                    os.makedirs(final_dirs)
                    evaluator.save_data(final_dirs)

                    end_time = time.time()
                    time_diff = end_time - start_time
                    minutes = int(time_diff / 60)
                    seconds = int(time_diff % 60)
                    model_dir = os.path.join(
                        self.base_path,
                        "seed_" + str(seed), 
                        self.organ_name, 
                        model_name, 
                        "train_size_" + str(len(train_indices)),
                        "num_samples_" + str(num_samples), 
                        "iteration_" + str(iteration)
                        )

                    torch.save(model_wrapper.model.state_dict(), model_dir + "/model.pth")
                    print(f"Time taken for latest model training/testing = {minutes}m {seconds}s")

                    with open(model_dir + "/training_x_embs.npy", "wb") as f:
                        np.save(f, np.array(model_wrapper.x_embs))

                    del model_wrapper      
                            
if __name__ == "__main__":
    tester = Tester()
    tester.main()
                            


