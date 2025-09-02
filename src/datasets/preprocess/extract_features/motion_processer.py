"""
Motion feature extractor 
"""
import os
import os.path as osp
import sys
import pickle
from omegaconf import OmegaConf

import torch

from PIL import Image
import numpy as np
import cv2
import imageio
import pickle
import time
from decord import VideoReader # must after import torch

from rich.progress import track




sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))))))
from src.datasets.preprocess.extract_features.face_segmentation import build_face_parser, get_face_mask, vis_parsing_maps
from src.thirdparty.liveportrait.src.utils.helper import load_model, concat_feat
from src.thirdparty.liveportrait.src.utils.io import load_image_rgb, resize_to_limit, load_video
from src.thirdparty.liveportrait.src.utils.video import get_fps, images2video, add_audio_to_video
from src.thirdparty.liveportrait.src.utils.camera import headpose_pred_to_degree, get_rotation_matrix

from src.thirdparty.liveportrait.src.utils.cropper import Cropper
from src.thirdparty.liveportrait.src.utils.crop import prepare_paste_back, paste_back, paste_back_with_face_mask
from src.thirdparty.liveportrait.src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from src.thirdparty.liveportrait.src.utils.helper import mkdir, basename, dct2device, is_image, calc_motion_multiplier
from src.utils.filter import smooth as ksmooth
from src.utils.filter import smooth_

from skimage.metrics import peak_signal_noise_ratio
import warnings


def psnr(imgs1, imgs2):
    psnrs = []
    for img1, img2 in zip(imgs1, imgs2):
        psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
        psnrs.append(psnr)
    return psnrs


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]

def dump(wfp, obj):
    wd = osp.split(wfp)[0]
    if wd != "" and not osp.exists(wd):
        mkdir(wd)

    _suffix = suffix(wfp)
    if _suffix == "npy":
        np.save(wfp, obj)
    elif _suffix == "pkl":
        pickle.dump(obj, open(wfp, "wb"))
    else:
        raise Exception("Unknown type: {}".format(_suffix))

def load(fp):
    suffix_ = suffix(fp)

    if suffix_ == "npy":
        return np.load(fp)
    elif suffix_ == "pkl":
        return pickle.load(open(fp, "rb"))
    else:
        raise Exception(f"Unknown type: {suffix}")


def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return osp.join(osp.dirname(filepath), basename(filepath))
    

class MotionProcesser(object):
    def __init__(self, cfg_path, device_id=0) -> None:
        device = f"cuda:{device_id}"
        cfg = OmegaConf.load(cfg_path)
        print(f"Load cfg from {osp.realpath(cfg_path)} done.")
        print(f"=============================== Driven CFG ===============================")
        print(OmegaConf.to_yaml(cfg))
        print(f"=============================== ========== ===============================")
        models_config = OmegaConf.load(cfg.models_config)

        # 1. init appearance feature extractor
        self.appearance_feature_extractor = load_model(
            cfg.appearance_feature_extractor_path, 
            models_config, 
            device, 
            'appearance_feature_extractor'
        )
        print(f'1. Load appearance_feature_extractor from {osp.realpath(cfg.appearance_feature_extractor_path)} done.')

        # 2. # init motion extractor
        self.motion_extractor = load_model(
            cfg.motion_extractor_path, 
            models_config, 
            device, 
            'motion_extractor'
        )
        print(f'2. Load motion_extractor from {osp.realpath(cfg.motion_extractor_path)} done.')
        
        # 3. init S and R
        if cfg.stitching_retargeting_module_path is not None and osp.exists(cfg.stitching_retargeting_module_path):
            self.stitching_retargeting_module = load_model(
                cfg.stitching_retargeting_module_path, 
                models_config, 
                device, 
                'stitching_retargeting_module'
            )
            print(f'3. Load stitching_retargeting_module from {osp.realpath(cfg.stitching_retargeting_module_path)} done.')
        else:
            self.stitching_retargeting_module = None
        
        # 4. init motion warper
        self.warping_module = load_model(
            cfg.warping_module_path, 
            models_config, 
            device, 
            'warping_module'
        )
        print(f"4. Load warping_module from {osp.realpath(cfg.warping_module_path)} done.")

        # 5. init decoder
        self.spade_generator = load_model(
            cfg.spade_generator_path, 
            models_config, 
            device, 
            'spade_generator'
        )
        print(f"Load generator from {osp.realpath(cfg.spade_generator_path)} done.")

        # # Optimize for inference
        self.compile = cfg.flag_do_torch_compile
        if self.compile:
            torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution
            self.warping_module = torch.compile(self.warping_module, mode='max-autotune')
            self.spade_generator = torch.compile(self.spade_generator, mode='max-autotune')

        # 6. init cropper
        crop_cfg = OmegaConf.load(cfg.crop_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg, image_type="human_face", device_id=device_id)
    
        self.cfg = cfg
        self.models_config = models_config
        self.device = device
    

        # 7. load crop mask
        self.mask_crop = cv2.imread(cfg.mask_crop, cv2.IMREAD_COLOR)
        # 8. load lib array
        with open(cfg.lip_array, 'rb') as f:
            self.lip_array = pickle.load(f)

        # 9. load face parser
        self.face_parser, self.to_tensor = build_face_parser(weight_path=cfg.face_parser_weight_path, resnet_weight_path=cfg.resnet_weight_path, device_id=device_id)

    def inference_ctx(self):    
        ctx = torch.autocast(device_type=self.device[:4], dtype=torch.float16,
                                 enabled=self.cfg.flag_use_half_precision)
        return ctx

    @torch.no_grad()
    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """ get the appearance feature of the image by F
        x: Bx3xHxW, normalized to 0~1
        """
        with self.inference_ctx():
            feature_3d = self.appearance_feature_extractor(x)

        return feature_3d.float()

    @torch.no_grad()
    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        with self.inference_ctx():
            kp_info = self.motion_extractor(x)

            if self.cfg.flag_use_half_precision:
                # float the dict
                for k, v in kp_info.items():
                    if isinstance(v, torch.Tensor):
                        kp_info[k] = v.float()

        return kp_info

    @torch.no_grad()
    def refine_kp(self, kp_info):
        bs = kp_info['exp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
        kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
        kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3
        if 'kp' in kp_info.keys():
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    @torch.no_grad()
    def transform_keypoint(self, kp_info: dict):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        kp: BxNx3
        """
        kp = kp_info['kp']    # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    @torch.no_grad()
    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        if self.stitching_retargeting_module is not None:
            bs, num_kp = kp_source.shape[:2]
            kp_driving_new = kp_driving.clone()
            # stich
            feat_stiching = concat_feat(kp_source, kp_driving_new)
            delta = self.stitching_retargeting_module['stitching'](feat_stiching) # Bxnum_kpx3

            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty

            return kp_driving_new

        return kp_driving

    @torch.no_grad()
    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> dict[str, torch.Tensor]:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """
        # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
        with self.inference_ctx():
            if self.compile:
                # Mark the beginning of a new CUDA Graph step
                torch.compiler.cudagraph_mark_step_begin()
            # get decoder input
            ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)

            # print(f"=============================================================================")
            # for out_key, out_value in ret_dct.items():
            #     if isinstance(out_value, str) or isinstance(out_value, int) or isinstance(out_value, float):
            #         print(f"{out_key}: {out_value}")
            #     elif isinstance(out_value, torch.Tensor):
            #         print(f"{out_key}: tensor shape {out_value.shape}, min: {torch.min(out_value)}, max: {torch.max(out_value)}, mean: {torch.mean(out_value)}, std: {torch.std(out_value)}")
            #     else:
            #         print(f"{out_key}: data type {type(out_value)}")
            # decode
            ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])

            # float the dict
            if self.cfg.flag_use_half_precision:
                for k, v in ret_dct.items():
                    if isinstance(v, torch.Tensor):
                        ret_dct[k] = v.float()

        return ret_dct
    
    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    @torch.no_grad()
    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(self.device)
        c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).to(self.device)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)
        return combined_eye_ratio_tensor

    @torch.no_grad()
    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(self.device)
        c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(self.device).reshape(1, 1) # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1) # 1x2
        return combined_lip_ratio_tensor

    def calc_ratio(self, lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    @torch.no_grad()
    def retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)

        delta = self.stitching_retargeting_module['lip'](feat_lip)

        return delta.reshape(-1, kp_source.shape[1], 3)

    @torch.no_grad()
    def retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)

        delta = self.stitching_retargeting_module['eye'](feat_eye)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def crop_image(self, img, do_crop=False):
        ######## process source info ########
        if do_crop:
            crop_info = self.cropper.crop_source_image(img, self.cropper.crop_cfg)
            if crop_info is None:
                raise Exception("No face detected in the source image!")
            lmk = crop_info['lmk_crop']
            img_crop_256x256 = crop_info['img_crop_256x256']
        else:
            crop_info = None
            lmk = self.cropper.calc_lmk_from_cropped_image(img)
            img_crop_256x256 = cv2.resize(img, (256, 256))  # force to resize to 256x256
        return img_crop_256x256, lmk, crop_info

    def crop_source_video(self, img_lst, do_crop=False):
        if do_crop:
            ret_s = self.cropper.crop_source_video(img_lst, self.cropper.crop_cfg)
            print(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
            img_crop_256x256_lst, lmk_crop_lst, M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
        else:
            M_c2o_lst = None
            lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(img_lst)
            img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in img_lst]  # force to resize to 256x256
        return img_crop_256x256_lst, lmk_crop_lst, M_c2o_lst
    
    def crop_driving_videos(self, img_lst, do_crop=False):
        if do_crop:
            ret_d = self.cropper.crop_driving_video(img_lst)
            print(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
            img_crop_lst, lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
            img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in img_lst]
        else:
            lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(img_lst)
            img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in img_lst]  # force to resize to 256x256
        return img_crop_256x256_lst, lmk_crop_lst

    def prepare_source(self, src_img):
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        # processing source image to tensor
        h, w = src_img.shape[:2]
        if h != self.cfg.input_height or w != self.cfg.input_width:
            x = cv2.resize(src_img, (self.cfg.input_width, self.cfg.input_height))
        else:
            x = src_img.copy()
        
        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(self.device)

        # extract features
        I_s = x
        f_s = self.extract_feature_3d(I_s)
        x_s_info = self.get_kp_info(I_s)
        
        return f_s, x_s_info
    
    def process_clips(self, clips):
        """ construct the input as standard
        clips: NxBxHxWx3, uint8
        """
        # resize to 256 x 256
        imgs = []
        for img in clips:
            h, w = img.shape[:2]
            if h != self.cfg.input_height or w != self.cfg.input_width:
                img = cv2.resize(img, (self.cfg.input_width, self.cfg.input_height))
            else:
                img = img.copy()
            imgs.append(img)

        # processing video frames to tensor
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        y = y.to(self.device)

        return y

    def prepare_driving_videos(self, vid_frames, feat_type="tensor"):
        """ get driving kp infos
        vid_frames: image list of HxWx3, uint8
        """
        # extract features
        total_len = len(vid_frames)
        kp_infos = {"pitch": [], "yaw": [], "roll": [], "t": [], "exp": [], "scale": [], "kp": []}
        for start_idx in range(0, total_len, self.cfg.batch_size):
            frames = vid_frames[start_idx: min(start_idx + self.cfg.batch_size, total_len)]
            frames = self.process_clips(frames).squeeze(1)
            kp_info = self.get_kp_info(frames)

            for k, v in kp_info.items():
                kp_infos[k].append(v)

        # combine the kp_infos
        for k, v in kp_infos.items():
            kp_infos[k] = torch.cat(v, dim=0)

        if feat_type == "np":
            for k, v in kp_infos.items():
                kp_infos[k] = v.cpu().numpy()

        return kp_infos

    def get_driving_template(self, kp_infos, smooth=False, dtype="pt_tensor"):
        kp_infos = self.refine_kp(kp_infos)
        motion_list = []
        n_frames = len(kp_infos["exp"])
        for idx in range(n_frames):
            exp = kp_infos["exp"][idx]
            scale = kp_infos["scale"][idx]
            t = kp_infos["t"][idx]
            pitch = kp_infos["pitch"][idx]
            yaw = kp_infos["yaw"][idx]
            roll = kp_infos["roll"][idx]
            
            R = get_rotation_matrix(pitch, yaw, roll)
            R = R.reshape(1, 3, 3)    
            
            exp = exp.reshape(1, 21, 3)
            scale = scale.reshape(1, 1)
            t = t.reshape(1, 3)
            pitch = pitch.reshape(1, 1)
            yaw = yaw.reshape(1, 1)
            roll = roll.reshape(1, 1)

            if dtype == "np":
                R = R.cpu().numpy().astype(np.float32)
                exp = exp.cpu().numpy().astype(np.float32)
                scale = scale.cpu().numpy().astype(np.float32)
                t = t.cpu().numpy().astype(np.float32)
                pitch = pitch.cpu().numpy().astype(np.float32)
                yaw = yaw.cpu().numpy().astype(np.float32)
                roll = roll.cpu().numpy().astype(np.float32)
            
            motion_list.append(
                {"exp": exp, "scale": scale, "R": R, "t": t, "pitch": pitch, "yaw": yaw, "roll": roll}
            )
        tgt_motion = {'n_frames': n_frames, 'output_fps': 25, 'motion': motion_list}

        if smooth:
            print("Smoothing motion sequence...")
            tgt_motion = smooth_(tgt_motion, method="ema")
        return tgt_motion

    @torch.no_grad()
    def update_delta_new_eyeball_direction(self, eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
        if eyeball_direction_x > 0:
                delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
                delta_new[0, 15, 0] += eyeball_direction_x * 0.001
        else:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.001
            delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

        delta_new[0, 11, 1] += eyeball_direction_y * -0.001
        delta_new[0, 15, 1] += eyeball_direction_y * -0.001
        blink = -eyeball_direction_y / 2.

        delta_new[0, 11, 1] += blink * -0.001
        delta_new[0, 13, 1] += blink * 0.0003
        delta_new[0, 15, 1] += blink * -0.001
        delta_new[0, 16, 1] += blink * 0.0003

        return delta_new

    def driven(self, f_s, x_s_info, s_lmk, c_s_eyes_lst, kp_infos, c_d_eyes_lst=None, c_d_lip_lst=None, smooth=False):
        # source kp info
        x_d_i_news=[]
        x_ss=[]
        f_ss=[]
        x_s_info = self.refine_kp(x_s_info)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        x_s = self.transform_keypoint(x_s_info)
        x_c_s = x_s_info["kp"]

        # driving kp infos
        driving_template_dct = self.get_driving_template(kp_infos, smooth)
        n_frames = driving_template_dct['n_frames']

        # driving params
        flag_normalize_lip = self.cfg.flag_normalize_lip
        flag_relative_motion = self.cfg.flag_relative_motion
        flag_source_video_eye_retargeting = self.cfg.flag_source_video_eye_retargeting
        lip_normalize_threshold = self.cfg.lip_normalize_threshold
        source_video_eye_retargeting_threshold = self.cfg.source_video_eye_retargeting_threshold
        animation_region = self.cfg.animation_region
        driving_option = self.cfg.driving_option
        flag_stitching = self.cfg.flag_stitching
        flag_eye_retargeting = self.cfg.flag_eye_retargeting
        flag_lip_retargeting = self.cfg.flag_lip_retargeting
        driving_multiplier = self.cfg.driving_multiplier
        lib_multiplier = self.cfg.lib_multiplier

        # let lip-open scalar to be 0 at first
        lip_delta_before_animation, eye_delta_before_animation = None, None
        if flag_normalize_lip and flag_relative_motion and s_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(c_d_lip_before_animation, s_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        # let eye-open scalar to be the same as the first frame if the latter is eye-open state
        if flag_source_video_eye_retargeting and s_lmk is not None:
            combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
            c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
            if c_d_eye_before_animation_frame_zero[0][0] < source_video_eye_retargeting_threshold:
                c_d_eye_before_animation_frame_zero = [[0.39]]
            combined_eye_ratio_tensor_before_animation = self.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, s_lmk)
            eye_delta_before_animation = self.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)
        
        # animate 
        I_p_lst = []
        for i in range(n_frames):
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, self.device)
            # R
            R_d_i = x_d_i_info['R']
            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()
            
            # enhance lip
            # if i > 0:
            #     for lip_idx in [6, 12, 14, 17, 19, 20]:
            #         x_d_i_info['exp'][:, lip_idx, :] = x_d_0_info['exp'][:, lip_idx, :] + (x_d_i_info['exp'][:, lip_idx, :] - x_d_0_info['exp'][:, lip_idx, :]) * lib_multiplier
            
            # normalize eye_ball, TODO
            x_d_i_info['exp'] = self.update_delta_new_eyeball_direction(0, -5, x_d_i_info['exp'])

            # debug
            #print(f"frame {i:03d}, src scale {x_s_info['scale']}, 0 scale {x_d_0_info['scale']}, i scale {x_d_i_info['scale']}")
            # delta
            delta_new = x_s_info['exp'].clone()
            if flag_relative_motion:
                # R
                if animation_region == "all" or animation_region == "pose":
                    R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s

                # exp
                if animation_region == "all" or animation_region == "exp":
                    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                elif animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
                elif animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, eyes_idx, :]
                
                # scale
                if animation_region == "all":
                    scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']

                # translation
                if animation_region == "all" or animation_region == "pose":
                    t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                # R
                if animation_region == "all" or animation_region == "pose":
                    R_new = R_d_i
                else:
                    R_new = R_s

                # exp
                if animation_region == "all" or animation_region == "exp":
                    EYE_IDX=[1,2,6,11,12,13,14,15,16,17,18,19,20]
                    delta_new[:, EYE_IDX, :] = x_d_i_info['exp'][:, EYE_IDX, :]
                    # for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    #     delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                    delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                    delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
                elif animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
                elif animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
                
                # scale
                scale_new = x_s_info['scale']

                # translation
                if animation_region == "all" or animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if flag_relative_motion and driving_option == "expression-friendly":
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s
            
            # Algorithm 1 in Liveportrait:
            if not flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if flag_eye_retargeting and s_lmk is not None and c_d_eyes_lst is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i, s_lmk)
                    eyes_delta = self.retarget_eye(x_s, combined_eye_ratio_tensor)

                if flag_lip_retargeting and s_lmk is not None and c_d_lip_lst is not None:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, s_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)

                if flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if flag_stitching:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
            x_d_i_news.append(x_d_i_new)
        f_s_s= f_s.expand(n_frames, *f_s.shape[1:]) 
        x_s_s = x_s.expand(n_frames, *x_s.shape[1:])  
        x_d_i_new = torch.cat(x_d_i_news, dim=0)        
        for start in range(0, n_frames, 100):
            end = min(start + 100,n_frames)
            with torch.no_grad(), torch.autocast('cuda'):
                out = self.warp_decode(f_s_s[start:end], x_s_s[start:end], x_d_i_new[start:end])        
                I_p_lst.append(out['out'])
        I_p=torch.cat(I_p_lst, dim=0) 
        I_p_i = self.parse_output(I_p)
        return I_p_i 

    def driven_debug(self, f_s, x_s_info, s_lmk, c_s_eyes_lst, driving_template_dct, c_d_eyes_lst=None, c_d_lip_lst=None):
        # source kp info
        x_s_info = self.refine_kp(x_s_info)
        x_c_s = x_s_info["kp"]
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        x_s = self.transform_keypoint(x_s_info)
        
        n_frames = driving_template_dct['n_frames']

        # driving params
        flag_normalize_lip = self.cfg.flag_normalize_lip
        flag_relative_motion = self.cfg.flag_relative_motion
        flag_source_video_eye_retargeting = self.cfg.flag_source_video_eye_retargeting
        lip_normalize_threshold = self.cfg.lip_normalize_threshold
        source_video_eye_retargeting_threshold = self.cfg.source_video_eye_retargeting_threshold
        animation_region = self.cfg.animation_region
        driving_option = self.cfg.driving_option
        flag_stitching = self.cfg.flag_stitching
        flag_eye_retargeting = self.cfg.flag_eye_retargeting
        flag_lip_retargeting = self.cfg.flag_lip_retargeting
        driving_multiplier = self.cfg.driving_multiplier

        # let lip-open scalar to be 0 at first
        lip_delta_before_animation, eye_delta_before_animation = None, None
        if flag_normalize_lip and flag_relative_motion and s_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(c_d_lip_before_animation, s_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        # let eye-open scalar to be the same as the first frame if the latter is eye-open state
        if flag_source_video_eye_retargeting and s_lmk is not None:
            combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
            c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
            if c_d_eye_before_animation_frame_zero[0][0] < source_video_eye_retargeting_threshold:
                c_d_eye_before_animation_frame_zero = [[0.39]]
            combined_eye_ratio_tensor_before_animation = self.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, s_lmk)
            eye_delta_before_animation = self.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)
        
        # animate 
        I_p_lst = []
        for i in range(n_frames):
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, self.device)
            # R
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys
            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()
            
            # debug
            #print(f"frame {i:03d}, src scale {x_s_info['scale']}, 0 scale {x_d_0_info['scale']}, i scale {x_d_i_info['scale']}")
            # delta
            delta_new = x_s_info['exp'].clone()
            if flag_relative_motion:
                # R
                if animation_region == "all" or animation_region == "pose":
                    R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s

                # exp
                if animation_region == "all" or animation_region == "exp":
                    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                elif animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
                elif animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, eyes_idx, :]
                
                # scale
                if animation_region == "all":
                    scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']

                # translation
                if animation_region == "all" or animation_region == "pose":
                    t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                # R
                if animation_region == "all" or animation_region == "pose":
                    R_new = R_d_i
                else:
                    R_new = R_s

                # exp
                if animation_region == "all" or animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                    delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                    delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
                elif animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
                elif animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
                
                # scale
                scale_new = x_s_info['scale']

                # translation
                if animation_region == "all" or animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if flag_relative_motion and driving_option == "expression-friendly":
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s
            
            # Algorithm 1 in Liveportrait:
            if not flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if flag_eye_retargeting and s_lmk is not None and c_d_eyes_lst is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i, s_lmk)
                    eyes_delta = self.retarget_eye(x_s, combined_eye_ratio_tensor)

                if flag_lip_retargeting and s_lmk is not None and c_d_lip_lst is not None:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, s_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)

                if flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if flag_stitching:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
            out = self.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)
        
        return I_p_lst 

    def read_image(self, image_path: str) -> list:
        img_rgb = load_image_rgb(image_path)
        img_rgb = resize_to_limit(img_rgb, self.cfg.source_max_dim, self.cfg.source_division)
        source_rgb_list = [img_rgb]
        print(f"Load image from {osp.realpath(image_path)} done.")
        return source_rgb_list

    def read_video(self, video_path: str, interval=None) -> list:
        vr = VideoReader(video_path)
        if interval is not None:
            video_frames = vr.get_batch(np.arange(0, len(vr), interval)).numpy()
        else:
            video_frames = [vr[0].numpy(), vr[len(vr) // 2].numpy(), vr[-1].numpy()]
        vr.seek(0)
        driving_rgb_list = []
        for video_frame in video_frames:
            # h, w = video_frame.shape[:2]
            # if h != self.cfg.output_height or w != self.cfg.output_width:
            #     video_frame = cv2.resize(video_frame, (self.cfg.output_height, self.cfg.output_width))
            driving_rgb_list.append(video_frame)

        return driving_rgb_list

    def prepare_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        y = y.to(self.device)

        return y

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.refine_kp(self.get_kp_info(I_i))
            x_s = self.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def load_template(self, wfp_template):
        print(f"Load from template: {wfp_template}, NOT the video, so the cropping video and audio are both NULL.")
        driving_template_dct = load(wfp_template)
        c_d_eyes_lst = driving_template_dct['c_eyes_lst'] if 'c_eyes_lst' in driving_template_dct.keys() else driving_template_dct['c_d_eyes_lst'] # compatible with previous keys
        c_d_lip_lst = driving_template_dct['c_lip_lst'] if 'c_lip_lst' in driving_template_dct.keys() else driving_template_dct['c_d_lip_lst']
        driving_n_frames = driving_template_dct['n_frames']
        flag_is_driving_video = True if driving_n_frames > 1 else False
        n_frames = driving_n_frames

        # set output_fps
        output_fps = driving_template_dct.get('output_fps', 25)
        print(f'The FPS of template: {output_fps}')
        return driving_template_dct

    def reconstruction(self, src_img, dst_imgs, video_path="template"):
        # prepare source
        src_img_256x256, s_lmk, _ = self.crop_image(src_img, do_crop=False)
        #c_s_eyes_lst, c_s_lip_lst = self.calc_ratio([s_lmk])
        c_s_eyes_lst = None
        f_s, x_s_info = self.prepare_source(src_img_256x256)
        
        # prepare driving videos
        dst_imgs_256x256, d_lmk_lst = self.crop_driving_videos(dst_imgs, do_crop=False)
        c_d_eyes_lst, c_d_lip_lst = self.calc_ratio(d_lmk_lst)
        kp_infos = self.prepare_driving_videos(dst_imgs_256x256)


        recs = self.driven(f_s, x_s_info, s_lmk, c_s_eyes_lst, kp_infos, c_d_eyes_lst, c_d_lip_lst)
        return recs

    def save_results(self, results, save_path, audio_path=None):
        save_dir = osp.dirname(save_path)
        save_name = osp.basename(save_path)
        final_video = osp.join(save_dir, f'final_{save_name}')

        images2video(results, wfp=save_path, fps=self.cfg.output_fps)

        if audio_path is not None:
            add_audio_to_video(save_path, audio_path, final_video)
            os.remove(save_path)
    
    def rec_score(self, video_path: str, interval=None, save_path=None):
        video_frames = self.read_video(video_path, interval=interval)
        #print(f"len frames: {len(video_frames)}, shape: {video_frames[0].shape}")
        recs = self.reconstruction(video_frames[0], video_frames[1:], video_path)
        if save_path is not None:
            self.save_results(recs, save_path)
        #print(f"len rec: {len(recs)}, shape: {recs[0].shape}")
        psnrs = psnr(video_frames[1:], recs)
        psnrs_np = np.array(psnrs)
        psnr_mean, psnr_std = np.mean(psnrs_np), np.std(psnrs_np)
        rec_score = {"mean": psnr_mean, "std": psnr_std}
        return rec_score

    @torch.no_grad()
    def paste_back_by_face_mask(self, result, crop_info, src_img, crop_src_image, use_laplacian=False):
        """
        paste back the result to the original image with face mask
        """
        # detect src mask
        crop_src_tensor = self.to_tensor(crop_src_image).unsqueeze(0).to(self.device)
        src_msks = get_face_mask(self.face_parser, crop_src_tensor)
        result_tensor = self.to_tensor(result).unsqueeze(0).to(self.device)
        result_msks = get_face_mask(self.face_parser, result_tensor)
        # combine masks
        masks = []
        for src_msk, result_msk in zip(src_msks, result_msks):
            mask = np.clip(src_msk + result_msk, 0, 1)
            masks.append(mask)
        result = paste_back_with_face_mask(result, crop_info, src_img, masks[0], use_laplacian=use_laplacian)
        return result

    def driven_by_audio(self, src_img, kp_infos, save_path, audio_path=None, smooth=False):
        # prepare source
        # prepare source
        src_img_256x256, s_lmk, crop_info = self.crop_image(src_img, do_crop=True)
        #c_s_eyes_lst, c_s_lip_lst = self.calc_ratio([s_lmk])
        c_s_eyes_lst = None
        f_s, x_s_info = self.prepare_source(src_img_256x256)

        mask_ori_float = prepare_paste_back(self.mask_crop, crop_info['M_c2o'], dsize=(src_img.shape[1], src_img.shape[0]))
        
        # prepare driving videos
        results = self.driven(f_s, x_s_info, s_lmk, c_s_eyes_lst, kp_infos, smooth=smooth)
        frames=results.shape[0]
        results = [paste_back(results[i], crop_info['M_c2o'], src_img, mask_ori_float) for i in range(frames)]
        self.save_results(results, save_path, audio_path)
    def mix_kp_infos(self, emo_kp_infos, lip_kp_infos, smooth=False, dtype="pt_tensor"):
        driving_emo_template_dct = self.get_driving_template(emo_kp_infos, smooth=False, dtype=dtype)
        if lip_kp_infos is not None:
            driving_lip_template_dct = self.get_driving_template(lip_kp_infos, smooth=smooth, dtype=dtype)
            driving_template_dct = {**driving_emo_template_dct}
            n_frames = min(driving_emo_template_dct['n_frames'], driving_lip_template_dct['n_frames'])
            driving_template_dct['n_frames'] = n_frames
            for i in range(n_frames):
                emo_motion = driving_emo_template_dct['motion'][i]['exp']
                lib_motion = driving_lip_template_dct['motion'][i]['exp']
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    emo_motion[:, lip_idx, :] = lib_motion[:, lip_idx, :]
                driving_template_dct['motion'][i]['exp'] = emo_motion
        else:
            driving_template_dct = driving_emo_template_dct
        
        return driving_template_dct

    def driven_by_mix(self, src_img, driving_video_path, kp_infos, save_path, audio_path=None, smooth=False):
        # prepare source
        src_img_256x256, s_lmk, crop_info = self.crop_image(src_img, do_crop=True)
        c_s_eyes_lst, c_s_lip_lst = self.calc_ratio([s_lmk])
        f_s, x_s_info = self.prepare_source(src_img_256x256)
        mask_ori_float = prepare_paste_back(self.mask_crop, crop_info['M_c2o'], dsize=(src_img.shape[1], src_img.shape[0]))
        # prepare driving videos
        driving_imgs = self.read_video(driving_video_path, interval=1)
        dst_imgs_256x256, d_lmk_lst = self.crop_driving_videos(driving_imgs, do_crop=True)
        c_d_eyes_lst, c_d_lip_lst = self.calc_ratio(d_lmk_lst)
        emo_kp_infos = self.prepare_driving_videos(dst_imgs_256x256)
        # mix kp_infos
        driving_template_dct = self.mix_kp_infos(emo_kp_infos, kp_infos, smooth=smooth)
        # driven 
        results = self.driven_debug(f_s, x_s_info, s_lmk, c_s_eyes_lst, driving_template_dct, c_d_eyes_lst=c_d_eyes_lst, c_d_lip_lst=c_d_lip_lst)
        results = [paste_back(result, crop_info['M_c2o'], src_img, mask_ori_float) for result in results]
        print(results.shape)
        self.save_results(results, save_path, audio_path)
    
    def drive_video_by_mix(self, video_path, driving_video_path, kp_infos, save_path, audio_path):
        # prepare driving videos
        driving_imgs = self.read_video(driving_video_path, interval=1)
        dst_imgs_256x256, d_lmk_lst = self.crop_driving_videos(driving_imgs, do_crop=True)
        emo_kp_infos = self.prepare_driving_videos(dst_imgs_256x256)
        # mix kp_infos
        #driving_template_dct = self.get_driving_template(emo_kp_infos, smooth=True, dtype="np")
        driving_template_dct = self.mix_kp_infos(emo_kp_infos, kp_infos, smooth=True, dtype="np")
        # driven
        self.video_lip_retargeting(
            video_path, None, 
            save_path, audio_path, 
            driving_template_dct=driving_template_dct, retargeting_ragion="exp"
        )

    def load_source_video(self, video_info, n_frames=-1):
        reader = imageio.get_reader(video_info, "ffmpeg")

        ret = []
        for idx, frame_rgb in enumerate(reader):
            if n_frames > 0 and idx >= n_frames:
                break
            ret.append(frame_rgb)

        reader.close()
        
        return ret

    def video_lip_retargeting(self, video_path, kp_infos, save_path, audio_path, c_d_eyes_lst=None, c_d_lip_lst=None, smooth=False, driving_template_dct=None, retargeting_ragion="exp"):
        # 0. process source motion template
        source_rgb_lst = load_video(video_path)
        source_rgb_lst = [resize_to_limit(img, self.cfg.source_max_dim, self.cfg.source_division) for img in source_rgb_lst]
        img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = self.crop_source_video(source_rgb_lst, do_crop=True)
        c_s_eyes_lst, c_s_lip_lst = self.calc_ratio(source_lmk_crop_lst)
        I_s_lst = self.prepare_videos(img_crop_256x256_lst)
        source_template_dct = self.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=25)
        # 1. prepare driving template
        if driving_template_dct is None:
            driving_template_dct = self.get_driving_template(kp_infos, smooth=smooth, dtype="np")
        # 2. driving
        n_frames = min(source_template_dct['n_frames'], driving_template_dct['n_frames'])
        # driving params
        I_p_lst = []
        I_p_pstbk_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = self.cfg.flag_normalize_lip
        flag_relative_motion = True #self.cfg.flag_relative_motion
        flag_source_video_eye_retargeting = self.cfg.flag_source_video_eye_retargeting
        lip_normalize_threshold = self.cfg.lip_normalize_threshold
        source_video_eye_retargeting_threshold = self.cfg.source_video_eye_retargeting_threshold
        animation_region = 'lip' #self.cfg.animation_region
        driving_option = self.cfg.driving_option
        flag_stitching = self.cfg.flag_stitching
        flag_eye_retargeting = self.cfg.flag_eye_retargeting
        flag_lip_retargeting = self.cfg.flag_lip_retargeting
        driving_multiplier = self.cfg.driving_multiplier
        driving_smooth_observation_variance = self.cfg.driving_smooth_observation_variance
        
        key_r = 'R' if 'R' in driving_template_dct['motion'][0].keys() else 'R_d'
        if flag_relative_motion:
            x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
            for i in range(n_frames):
                for idx in [6, 12, 14, 17, 19, 20]:
                    # lip motion use abs motion
                    x_d_exp_lst[i][:, idx, :] = driving_template_dct['motion'][i]['exp'][:, idx, :]
            x_d_exp_lst_smooth = ksmooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, self.device, driving_smooth_observation_variance)
            
            if animation_region == "all" or animation_region == "pose" or "all" in animation_region:
                x_d_r_lst = [(np.dot(driving_template_dct['motion'][i][key_r], driving_template_dct['motion'][0][key_r].transpose(0, 2, 1))) @ source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                x_d_r_lst_smooth = ksmooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, self.device, driving_smooth_observation_variance)
        else:
            x_d_exp_lst = [driving_template_dct['motion'][i]['exp'] for i in range(n_frames)]
            x_d_exp_lst_smooth = ksmooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, self.device, driving_smooth_observation_variance)

            if animation_region == "all" or animation_region == "pose" or "all" in animation_region:
                x_d_r_lst = [driving_template_dct['motion'][i][key_r] for i in range(n_frames)]
                x_d_r_lst_smooth = ksmooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, self.device, driving_smooth_observation_variance)
        
        # driving all
        for i in track(range(n_frames), description='🚀Retargeting...', total=n_frames):
            x_s_info = source_template_dct['motion'][i]
            x_s_info = dct2device(x_s_info, self.device)

            source_lmk = source_lmk_crop_lst[i]
            img_crop_256x256 = img_crop_256x256_lst[i]
            I_s = I_s_lst[i]
            f_s = self.extract_feature_3d(I_s)

            x_c_s = x_s_info['kp']
            R_s = x_s_info['R']
            x_s =x_s_info['x_s']

            # let lip-open scalar to be 0 at first if the input is a video
            lip_delta_before_animation = None
            if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                    lip_delta_before_animation = self.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                else:
                    lip_delta_before_animation = None

            # let eye-open scalar to be the same as the first frame if the latter is eye-open state
            eye_delta_before_animation = None
            if flag_source_video_eye_retargeting and source_lmk is not None:
                if i == 0:
                    combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
                    c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                    if c_d_eye_before_animation_frame_zero[0][0] < source_video_eye_retargeting_threshold:
                        c_d_eye_before_animation_frame_zero = [[0.39]]
                combined_eye_ratio_tensor_before_animation = self.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, source_lmk)
                eye_delta_before_animation = self.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)
            
            if flag_stitching:  # prepare for paste back
                mask_ori_float = prepare_paste_back(self.mask_crop, source_M_c2o_lst[i], dsize=(source_rgb_lst[i].shape[1], source_rgb_lst[i].shape[0]))
            
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, self.device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()
            
            delta_new = x_s_info['exp'].clone()
            if flag_relative_motion:
                if animation_region == "all" or animation_region == "pose" or "all" in animation_region:
                    R_new = x_d_r_lst_smooth[i]
                else:
                    R_new = R_s
                if animation_region == "all" or animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                elif animation_region == "all_wo_lip" or animation_region == "exp_wo_lip":
                    for idx in [1, 2, 11, 13, 15, 16, 18]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                elif animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :]
                elif animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :]
                
                scale_new = x_s_info['scale']
                t_new = x_s_info['t']
            else:
                if animation_region == "all" or animation_region == "pose" or "all" in animation_region:
                    R_new = x_d_r_lst_smooth[i] 
                else:
                    R_new = R_s
                if animation_region == "all" or animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] 
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2] 
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2] 
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                elif animation_region == "all_wo_lip" or animation_region == "exp_wo_lip":
                    for idx in [1, 2, 11, 13, 15, 16, 18]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                elif animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :]
                elif animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :]
                scale_new = x_s_info['scale']
                if animation_region == "all" or animation_region == "pose" or "all" in animation_region:
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # Algorithm 1:
            if not flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if flag_eye_retargeting and source_lmk is not None and c_d_eyes_lst is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.retarget_eye(x_s, combined_eye_ratio_tensor)
                if flag_lip_retargeting and source_lmk is not None and c_d_lip_lst is not None:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)

                if flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if flag_stitching:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
            out = self.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                #I_p_pstbk = self.paste_back_by_face_mask(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], img_crop_256x256, use_laplacian=True)
                I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_float, use_laplacian=True)
                I_p_pstbk_lst.append(I_p_pstbk)
            
        if len(I_p_pstbk_lst) > 0:
            self.save_results(I_p_pstbk_lst, save_path, audio_path)
        else:
            self.save_results(I_p_lst, save_path, audio_path)

    @torch.no_grad()
    def video_reconstruction_test(self, video_tensor, xs, save_path):
        # video_tensor, (1, F, C, H, W), [-1, 1]
        # xs, (1, F, 63)
        result_lst = []
        #ori_videos = []
        video_tensor = video_tensor[0:1] * 0.5 + 0.5  # [-1, 1] -> [0, 1], 1xTx3xHxW
        video_tensor = torch.clip(video_tensor, 0, 1)
        video_tensor = video_tensor.permute(1, 0, 2, 3, 4) # 1xTx3xHxW -> Tx1x3xHxW
        video = video_tensor.to(self.device)
        xs = xs[0:1].permute(1, 0, 2)    # 1xTx63 -> Tx1x63
        xs = xs.reshape(-1, 1, 21, 3)
        xs = xs.to(self.device)

        x_s_0 = xs[0]
        I_s_0 = torch.nn.functional.interpolate(video[0], size=(256, 256), mode='bilinear')
        f_s_0 = self.extract_feature_3d(I_s_0)

        for i in range(video_tensor.shape[0]):
            #I_s = video[i]   # 1x3xHxW
            #ori_videos.append((I_s.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
            x_s = self.stitching(x_s_0, xs[i])
            out = self.warp_decode(f_s_0, x_s_0, x_s)
            I_p_i = self.parse_output(out['out'])[0]
            result_lst.append(I_p_i)

        #save_dir = osp.dirname(save_path)
        #ori_path = osp.join(save_dir, "ori.mp4")
        #save_path = osp.join(save_dir, "rec.mp4")
        self.save_results(result_lst, save_path, audio_path=None)
        #self.save_results(ori_videos, ori_path, audio_path=None)

    @torch.no_grad()
    def self_driven(self, image_tensor, xs, save_path, length):
        result_lst = []
        image_tensor = image_tensor[0:1] * 0.5 + 0.5    # [-1, 1] -> [0, 1], 1x3xHxW
        image_tensor = torch.clip(image_tensor, 0, 1)
        image = image_tensor.to(self.device)
        I_s_0 = torch.nn.functional.interpolate(image, size=(256, 256), mode='bilinear')

        xs = xs[0:1].permute(1, 0, 2)    # 1xTx63 -> Tx1x63
        xs = xs.reshape(-1, 1, 21, 3)
        xs = xs.to(self.device)

        x_s_0 = xs[0]
        f_s_0 = self.extract_feature_3d(I_s_0)

        for i in range(xs.shape[0]):
            x_d = self.stitching(x_s_0, xs[i])
            out = self.warp_decode(f_s_0, x_s_0, x_d)
            I_p_i = self.parse_output(out['out'])[0]
            result_lst.append(I_p_i)

        assert len(result_lst) == length, f"length of result_lst is {len(result_lst)}, but length is {length}"

        self.save_results(result_lst, save_path, audio_path=None)


