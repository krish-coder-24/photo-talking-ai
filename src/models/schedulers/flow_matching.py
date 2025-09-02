# modified from https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py#L23

import torch
import torch.nn as nn
import numpy as np


class ModelSamplingDiscreteFlow(nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, num_train_timesteps=1000, shift=1.0, **kwargs):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        ts = self.to_sigma(torch.arange(1, num_train_timesteps + 1, 1))  # [1/1000, 1]
        self.register_buffer("sigmas", ts)
    
    @property
    def sigma_min(self):
        return self.sigmas[0]
    
    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def to_timestep(self, sigma):
        return sigma * self.num_train_timesteps
    
    def to_sigma(self, timestep: torch.Tensor):
        timestep = timestep / self.num_train_timesteps
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)
    
    def uniform_sample_t(self, batch_size, device):
        ts = (self.sigma_max - self.sigma_min) * torch.rand(batch_size, device=device) + self.sigma_min
        return ts

    def calculate_denoised(self, sigma, model_output, model_input):
        # model ouput, vector field, v = dx = (x_1 - x_0)
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image):
        return sigma * noise + (1.0 - sigma) * latent_image
        
    def add_noise(self, sample, noise=None, timesteps=None):
        # sample, B, L, D
        if timesteps is None:
            # Sample time step
            batch_size = sample.shape[0]
            sigmas = self.uniform_sample_t(batch_size, device=sample.device).to(dtype=sample.dtype)  # (B,)
            timesteps = self.to_timestep(sigmas)
        else:
            timesteps = timesteps.to(device=sample.device, dtype=sample.dtype)
            sigmas = self.to_sigma(timesteps)
        
        sigmas = sigmas.view(-1, 1, 1)            # (B, 1, 1)
        noise = torch.randn_like(sample)
        noisy_samples = sigmas * noise + (1.0 - sigmas) * sample
        return noisy_samples, noise, noise - sample, timesteps

    def set_timesteps(self, num_inference_steps, device=None):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        start = self.to_timestep(self.sigma_max)
        end = self.to_timestep(self.sigma_min)
        timesteps = torch.linspace(start, end, num_inference_steps)

        self.timesteps = torch.from_numpy(np.array(timesteps)).to(device)

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        return x[(...,) + (None,) * dims_to_append]

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    @torch.no_grad()
    def step(self, model_output, timestep, sample, method="euler", **kwargs):
        """
        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model, direction (noise - x_0).
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process, x_t.
            method (`str`):
                ODE solver, `euler` or `dpmpp_2m`

        Returns:
            `tuple`:
                the sample tensor.
        """

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        sigma = self.to_sigma(timestep)
        prev_sigma = sigma - (self.sigma_max - self.sigma_min) / (self.num_inference_steps - 1)
        prev_sigma = 0.0 if prev_sigma < 0.0 else prev_sigma

        if method == "euler":
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            dt = prev_sigma - sigma
            prev_sample = sample + model_output * dt
        elif method == "dpmpp_2m":
            """DPM-Solver++(2M)."""
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported ode solver: {method}, only supports `euler` or `dpmpp_2m`")

        pred_original_sample = sample - model_output * sigma

        return (
            prev_sample,
            pred_original_sample
        )

    def get_pred_original_sample(self, model_output, timestep, sample):
        sigma = self.to_sigma(timestep).view(-1, 1, 1)
        pred_original_sample = sample - model_output * sigma

        return pred_original_sample