# encoding = 'utf-8'
import os.path as osp

import math
from rich.progress import track

from omegaconf import OmegaConf

import torch
import torch.nn as nn

from .talking_head_dit import TalkingHeadDiT_models
import sys
from ..schedulers.scheduling_ddim import DDIMScheduler
from ..schedulers.flow_matching import ModelSamplingDiscreteFlow
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
scheduler_map = {
    "ddim": DDIMScheduler,
    # "ddpm": DiffusionSchedule,
    "flow_matching": ModelSamplingDiscreteFlow
}
lip_dims=[18, 19, 20, 36, 37, 38, 42, 43, 44, 51, 52, 53, 57, 58, 59, 60, 61, 62]

class MotionDiffusion(nn.Module):
    def __init__(self, config, device="cuda", dtype=torch.float32, smo_wsize=3, loss_type="l2"):
        super().__init__()

        self.config = config
        self.smo_wsize = smo_wsize
        print(f"================================== Init Motion GeneratorV2 ==================================")
        print(OmegaConf.to_yaml(self.config))
        
        motion_gen_config = config.motion_generator
        motion_gen_params = motion_gen_config.params
        
        audio_proj_config = config.audio_projector
        audio_proj_params = audio_proj_config.params
        
        scheduler_config = config.noise_scheduler
        scheduler_params = scheduler_config.params

        self.device = device

        # init motion generator
        self.talking_head_dit = TalkingHeadDiT_models[config.model_name](
            input_dim           = motion_gen_params.input_dim * 2,
            output_dim          = motion_gen_params.output_dim,
            seq_len             = motion_gen_params.n_pred_frames,
            audio_unit_len      = audio_proj_params.sequence_length,
            audio_blocks        = audio_proj_params.blocks,
            audio_dim           = audio_proj_params.audio_feat_dim,
            audio_tokens        = audio_proj_params.context_tokens,
            audio_embedder_type = audio_proj_params.audio_embedder_type,
            audio_cond_dim      = audio_proj_params.audio_cond_dim,
            norm_type           = motion_gen_params.norm_type,
            qk_norm             = motion_gen_params.qk_norm,
            exp_dim             = motion_gen_params.exp_dim
        )
        self.input_dim = motion_gen_params.input_dim
        self.exp_dim = motion_gen_params.exp_dim

        self.audio_feat_dim = audio_proj_params.audio_feat_dim 
        self.audio_seq_len = audio_proj_params.sequence_length
        self.audio_blocks = audio_proj_params.blocks
        self.audio_margin = (audio_proj_params.sequence_length - 1) // 2
        self.indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        ).unsqueeze(0)  # Generates [-2, -1, 0, 1, 2], size 1 x (2*self.audio_margin+1)
        
        self.n_prev_frames = motion_gen_params.n_prev_frames
        self.n_pred_frames = motion_gen_params.n_pred_frames
        
        # init diffusion schedule
        self.scheduler = scheduler_map[scheduler_config.type](
            num_train_timesteps = scheduler_params.num_train_timesteps,
            beta_start          = scheduler_params.beta_start, 
            beta_end            = scheduler_params.beta_end, 
            beta_schedule       = scheduler_params.mode,
            prediction_type     = scheduler_config.sample_mode,
            time_shifting       = scheduler_params.time_shifting,
        )
        self.scheduler_type = scheduler_config.type
        self.eta = scheduler_params.eta
        self.scheduler.set_timesteps(scheduler_params.num_inference_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps
        print(f"time steps: {self.timesteps}")
        
        self.sample_mode = scheduler_config.sample_mode
        assert (self.sample_mode in ["noise", "sample"], f"Unknown sample mode {self.sample_mode}, should be noise or sample")

        # init other params
        self.audio_drop_ratio = config.train.audio_drop_ratio
        self.pre_drop_ratio = config.train.pre_drop_ratio

        self.null_audio_feat = nn.Parameter(
            torch.randn(1, 1, 1, 1, self.audio_feat_dim), 
            requires_grad=True
        ).to(device=self.device, dtype=dtype)

        self.null_motion_feat = nn.Parameter(
            torch.randn(1, 1, self.input_dim),
            requires_grad=True
        ).to(device=self.device, dtype=dtype)
        
        # for segments fusion
        self.overlap_len = min(16, self.n_pred_frames - 16)
        self.fuse_alpha = torch.arange(self.overlap_len, device=self.device, dtype=dtype).reshape(1, -1, 1) / self.overlap_len

        self.dtype = dtype
        self.loss_type = loss_type

        total_params = sum(p.numel() for p in self.parameters())
        print('Number of parameter: % .4fM' % (total_params / 1e6))
        print(f"================================== init Motion GeneratorV2: Done ==================================")
        
    def _smooth(self, motion):
        # motion, B x L x D
        if self.smo_wsize <= 1:
            return motion
        new_motion = motion.clone()
        n = motion.shape[1]
        half_k = self.smo_wsize // 2
        for i in range(n):
            ss = max(0, i - half_k)
            ee = min(n, i + half_k + 1)
            # only smooth head pose motion
            motion[:, i, self.exp_dim:] = torch.mean(new_motion[:, ss:ee, self.exp_dim:], dim=1)
            
        return motion

    def _fuse(self, prev_motion, cur_motion):
        r1 = prev_motion[:, -self.overlap_len:]
        r2 = cur_motion[:, :self.overlap_len]
        r_fuse = r1 * (1 - self.fuse_alpha) + r2 * self.fuse_alpha

        prev_motion[:, -self.overlap_len:] = r_fuse    # fuse last
        return prev_motion
    
    @torch.no_grad()
    def sample_subclip(
        self, 
        audio, 
        ref_kp,
        prev_motion,
        emo=None,
        cfg_scale=1.15, 
        init_latents=None,
        dynamic_threshold = None
    ):
        # prepare audio feat
        batch_size = audio.shape[0]
        audio = audio.to(self.device)
        if audio.ndim == 4:
            audio = audio.unsqueeze(2)
        
        # reference keypoints
        ref_kp = ref_kp.view(batch_size, 1, -1)
        
        # cfg
        if cfg_scale > 1:
            uncond_audio = self.null_audio_feat.expand(
               batch_size, self.n_pred_frames, self.audio_seq_len, self.audio_blocks, -1
            )
            audio = torch.cat([uncond_audio,audio], dim=0)
            ref_kp = torch.cat([ref_kp] * 2, dim=0)
            if emo is not None:
                uncond_emo = torch.Tensor([self.talking_head_dit.num_emo_class]).long().to(self.device)
                emo = torch.cat([uncond_emo,emo], dim=0)
        ref_kp = ref_kp.repeat(1, audio.shape[1], 1)  # B, L, kD

        # prepare noisy motion
        if init_latents is None:
            latents = torch.randn((batch_size, self.n_pred_frames, self.input_dim)).to(self.device)
        else:
            latents = init_latents
        
        prev_motion = prev_motion.expand_as(latents).to(dtype=self.dtype)
        latents = latents.to(dtype=self.dtype)
        audio = audio.to(dtype=self.dtype)
        ref_kp = ref_kp.to(dtype=self.dtype)
        for t in track(self.timesteps, description='🚀Denosing', total=len(self.timesteps)):
            motion_in = torch.cat([prev_motion, latents], dim=-1)
            step_in = torch.tensor([t] * batch_size, device=self.device, dtype=self.dtype)
            if cfg_scale > 1:
                motion_in = torch.cat([motion_in] * 2, dim=0)
                step_in = torch.cat([step_in] * 2, dim=0)
            # predict
            pred = self.talking_head_dit(
                motion     = motion_in, 
                times       = step_in,
                audio      = audio,
                emo        = emo,
                audio_cond = ref_kp
            )

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = pred.reshape(batch_size * 2, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                pred = torch.clamp(pred, min=-s, max=s)

            # CFG
            if cfg_scale > 1:
                # uncond_pred, emo_cond_pred, all_cond_pred = pred.chunk(3, dim=0)
                # pred = uncond_pred + 8 * (emo_cond_pred - uncond_pred) + 1.2 * (all_cond_pred - emo_cond_pred)
                uncond_pred, cond_pred = pred.chunk(2, dim=0)
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            # Step
            latents = self.scheduler.step(pred, t, latents, eta=self.eta, return_dict=False)[0]
        self.talking_head_dit.bank=[]
        return latents
            
    @torch.no_grad()
    def sample(self, audio, ref_kp, prev_motion, cfg_scale=1.15, audio_pad_mode="zero", emo=None,dynamic_threshold=None):
        # prev_motion, B, 1, D
        # for inference with any length audio
        # crop audio into n_subdivision according to n_pred_frames 
        clip_len = audio.shape[0]
        stride = self.n_pred_frames - self.overlap_len
        if clip_len <= self.n_pred_frames:
            n_subdivision = 1
        else:
            n_subdivision = math.ceil((clip_len - self.n_pred_frames) / stride) + 1
        
        # padding
        n_padding_frames = self.n_pred_frames + stride * (n_subdivision - 1) - clip_len
        if n_padding_frames > 0:
            padding_value = 0
            if audio_pad_mode == 'zero':
                padding_value = torch.zeros_like(audio[-1:])
            elif audio_pad_mode == 'replicate':
                padding_value = audio[-1:]
            else:
                raise ValueError(f'Unknown pad mode: {audio_pad_mode}')
            audio = torch.cat(
                [audio[:1]] * self.audio_margin \
                + [audio] + [padding_value] * n_padding_frames \
                + [audio[-1:]] * self.audio_margin, 
                dim=0
            )
        
        center_indices = torch.arange(
            self.audio_margin,
            audio.shape[0] - self.audio_margin
        ).unsqueeze(1) + self.indices
        audio_tensor = audio[center_indices]   # T, L, b, aD

        # add reference keypoints
        motion_lst = []
        #init_latents = torch.randn((1, self.n_pred_frames, self.motion_dim)).to(device=self.device)
        init_latents = None
        # emotion label
        if emo is not None:
            emo = torch.Tensor([emo]).long().to(self.device)
        start_idx = 0
        for i in range(0, n_subdivision):
            print(f"Sample subclip {i+1}/{n_subdivision}")
            end_idx = start_idx + self.n_pred_frames
            audio_segment = audio_tensor[start_idx: end_idx].unsqueeze(0)
            start_idx += stride

            # debug
            #print(f"scale:")
            
            motion_segment = self.sample_subclip(
                audio             = audio_segment, 
                ref_kp            = ref_kp,
                prev_motion       = prev_motion,
                emo               = emo,
                cfg_scale         = cfg_scale,
                init_latents      = init_latents,
                dynamic_threshold = dynamic_threshold
            )
            # smooth

            motion_segment = self._smooth(motion_segment)
            # update prev motion
            prev_motion = motion_segment[:, stride-1:stride].clone()

            # save results
            motion_coef = motion_segment
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames
            
            if len(motion_lst) > 0:
                # fuse segments
                motion_lst[-1] = self._fuse(motion_lst[-1], motion_coef)
                motion_lst.append(motion_coef[:, self.overlap_len:])
            else:
                motion_lst.append(motion_coef)
                
        motion = torch.cat(motion_lst, dim=1)
        # smooth for full clip
        motion = self._smooth(motion)
        motion = motion.squeeze()
        return motion.float()


    