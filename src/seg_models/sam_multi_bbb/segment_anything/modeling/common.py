# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import Parameter

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class BayesBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        mlp_dim: int, 
        act = nn.GELU,
        skip_connect: bool = True,
        device: str = "cuda:0"
    ) -> None:
        super().__init__()
        
        self.skip_connect = True
        self.activation = act()
        self.fc1 = LinearBayes(embedding_dim, mlp_dim, device=device)
        self.fc2 = LinearBayes(mlp_dim, embedding_dim, device=device)

    def forward(
        self, 
        x: torch.Tensor,
        stochastic: bool = True
    ) -> torch.Tensor:

        kl_total = 0.0

        x0 = torch.clone(x)
        x, kl = self.fc1(x, stochastic)
        kl_total += kl
        x = self.activation(x)
        x, kl = self.fc2(x, stochastic)
        kl_total += kl

        if self.skip_connect:
            x = x + x0
            
        return x, kl_total


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class LinearBayes(nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            use_bias: bool = True,
            device: str = "cuda:0"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.use_bias = use_bias

        priors = {
            "prior_mu": 0.,
            "prior_sigma": 1,
            "posterior_mu_init": (0, 0.1),
            "posterior_rho_init": (-3, 0.1)
        }

        self.prior_mu = priors["prior_mu"]
        self.prior_sigma = priors["prior_sigma"]
        self.posterior_mu_init = priors["posterior_mu_init"]
        self.posterior_rho_init = priors["posterior_rho_init"]

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_init)
        self.W_rho.data.normal_(*self.posterior_rho_init)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_init)
            self.bias_rho.data.normal_(*self.posterior_rho_init)

    def forward(
            self, 
            x,
            stochastic=True
    ):
        if stochastic:
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            W_eps = torch.Tensor(self.W_mu.size()).normal_().to(self.device)
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_eps = torch.Tensor(self.bias_mu.size()).normal_().to(self.device)
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
            
            kl = self.kl_loss()
     
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None
            kl = 0.0
     
        return nn.functional.linear(x, weight, bias=bias), kl
    
    def calculate_kl(self, mu_q, sig_q, mu_p, sig_p):
        kl = torch.sum(torch.log(sig_q / sig_p) + ((sig_p.pow(2) + (mu_p - mu_q).pow(2))/(2 * sig_q**2)) - 0.5)
        return kl
    
    def kl_loss(self):
        kl = self.calculate_kl(
            self.prior_mu,
            self.prior_sigma,
            self.W_mu, 
            self.W_sigma
            )
        return kl
