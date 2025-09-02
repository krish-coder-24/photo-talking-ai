
import os
from posixpath import isfile
from re import A
import sys
import os.path as osp

from typing import List, Dict, Tuple, Optional, Union, Any

import yaml
from omegaconf import OmegaConf

import math
import librosa
import soundfile
import numpy as np

from einops import rearrange

import torch
import torch.nn.functional as F

from pydub import AudioSegment
from audio_separator.separator import Separator

from transformers import Wav2Vec2FeatureExtractor, HubertModel

from src.utils.rprint import rlog as log
from src.utils.util import resample_audio

from src.models.audio.wav2vec_modified import Wav2VecModel
from src.models.audio.hubert import HubertModel_ as HubertModel


def pad_audio(audio, audio_unit=320, pad_threshold=80):
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')

    return audio


def cut_audio(audio_path: str, save_dir: str, length=60) -> List[str]:
    """Cut audio into sub-divisions and return subfile paths. Supports wav format.

    Args:
        audio_path (str): the source audio file path
        save_dir (str): the save directory of sub-divisions
        length (int, optional): The max length of each sub-division. Defaults to 60 secs.

    Returns:
        List[str]: the subfile paths
    """
    audio_name = osp.basename(audio_path).split('.')[0]
    audio = AudioSegment.from_wav(audio_path)
    segment_length = length * 1000. # pydub uses milliseconds
    num_segments = math.ceil(len(audio) / segment_length)
    
    os.makedirs(save_dir, exist_ok=True)
    audio_list = []
    
    if num_segments > 1:
        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, len(audio))
            segment = audio[start_time:end_time]
            
            path = osp.join(save_dir, f"{audio_name}_segment_{i+1}.wav")
            audio_list.append(path)
            segment.export(path, format="wav")
    else:
        audio_list = [audio_path]
    return audio_list
    
    
class AudioProcessor(object):
    def __init__(self, cfg_path: str, is_training: bool = False, device_id=0) -> None:
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.is_training = is_training
        log("========================================= Audio Processer =========================================")
        log(OmegaConf.to_yaml(cfg))

        # setting device 
        self.device_id = device_id
        self.use_half = cfg.device_params.flag_use_half_precision
        if cfg.device_params.flag_force_cpu:
            self.device = 'cpu'
        else:
            try:
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cuda:' + str(self.device_id)
            except:
                self.device = 'cuda:' + str(self.device_id)

        # init audio separator
        self.audio_separator = None
        self.cache_dir = cfg.cache_dir
        self.tmp_dir = cfg.tmp_dir
        self.use_audio_separator = cfg.model_params.use_audio_separator
        self.audio_separator_name = cfg.model_params.audio_separator_name
        self.audio_separator_path = cfg.model_weights.audio_separator_path
        self.set_audio_separator(cfg.cache_dir)
        
        # load audio encoder, wav2vec or hubert
        self.model_name = cfg.model_params.model_name
        self.is_chinese = cfg.model_params.is_chinese
        self.audio_encoder, self.feature_extractor = self.load_model(
            model_name = cfg.model_params.model_name, 
            model_type = cfg.model_params.model_type, 
            is_chinese = cfg.model_params.is_chinese, 
        )
        self.only_last_features = cfg.model_params.only_last_features
        if cfg.model_params.only_last_features:
            self.feature_shape = (1, 768)
        else:
            self.feature_shape = (12, 768)     # features of 12 blocks
        
        # init data params
        self.sample_strategy = cfg.data_params.sample_strategy
        self.sample_rate = cfg.data_params.sample_rate
        self.fps = cfg.data_params.fps
        self.audio_unit = cfg.data_params.sample_rate / cfg.data_params.fps   # num of audio samples per frame
        self.max_length = cfg.data_params.max_length
        self.subclip_len = cfg.data_params.sub_clip_length
        self.save_to_cpu = cfg.data_params.save_to_cpu
        self.pad_mode = cfg.data_params.audio_pad_mode

        log("========================================= Audio Processer: Done =========================================")
        
    def load_model(self, model_name: str="wav2vec", model_type: str="base", is_chinese: bool = False):
        assert model_name in ["wav2vec", "hubert"], f"Unknown audio model {model_name}, only support wav2vec or hubert"
        assert model_type in ["base", "large"], f"Unknown audio model type {model_type}, only support base or large"

        if model_name == "wav2vec":
            # load wav2vec model weights
            if is_chinese:
                if model_type == "base":
                    model_weight_path = self.cfg.model_weights.wav2vec_path.chinese.base
                else:
                    model_weight_path = self.cfg.model_weights.wav2vec_path.chinese.large
            else:
                if model_type == "base":
                    model_weight_path = self.cfg.model_weights.wav2vec_path.default.base
                else:
                    model_weight_path = self.cfg.model_weights.wav2vec_path.default.large
            if model_weight_path is None:
                raise ValueError(f"model_weight_path is None")
            audio_encoder = Wav2VecModel.from_pretrained(model_weight_path, local_files_only=True).to(device=self.device)
        else:
            if is_chinese:
                if model_type == "base":
                    model_weight_path = self.cfg.model_weights.hubert_path.chinese.base
                else:
                    model_weight_path = self.cfg.model_weights.hubert_path.chinese.large
            else:
                if model_type == "base":
                    model_weight_path = self.cfg.model_weights.hubert_path.default.base
                else:
                    model_weight_path = self.cfg.model_weights.hubert_path.default.large
            if model_weight_path is None:
                raise ValueError(f"model_weight_path is None")
            audio_encoder = HubertModel.from_pretrained(model_weight_path, local_files_only=True).to(device=self.device)

        log(f"{model_name}-{model_type}-chinese-{is_chinese} model has beed loaded from {model_weight_path}")
        total_params = sum(p.numel() for p in audio_encoder.parameters())
        print('Number of parameter: % .4fM' % (total_params / 1e6))
        
        # weights initialization
        audio_encoder.feature_extractor._freeze_parameters()
        if not self.cfg.model_params.is_original:
            frozen_layers = [0, 1]
            for name, param in audio_encoder.named_parameters():
                if name.startswith("feature_projection"):
                    param.requires_grad = False
                if name.startswith("encoder.layers"):
                    layer = int(name.split(".")[2])
                    if layer in frozen_layers:
                        param.requires_grad = False

        audio_encoder = audio_encoder.to(self.device)
        if self.use_half:
            audio_encoder = audio_encoder.half()
        audio_encoder.eval()

        # feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_weight_path)

        return audio_encoder, feature_extractor

    def set_audio_separator(self, output_dir: str) -> None:
        del self.audio_separator
        
        if self.audio_separator_name is not None and self.use_audio_separator:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as _:
                print("Fail to create the output cache dir.")
            self.audio_separator = Separator(
                output_dir=output_dir,
                output_single_stem="vocals",
                model_file_dir=self.audio_separator_path,
            )
            self.audio_separator.load_model(self.audio_separator_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."
        else:
            self.audio_separator=None
            log("Use audio directly without vocals seperator.")
    
    def seperate_audio(self, audio_path: str, output_dir: Union[str, None] = None) -> str:
        if output_dir is not None:
            if output_dir != self.cache_dir:
                # reload audio separator
                self.set_audio_separator(output_dir)
        
        if self.audio_separator is not None:
            # 1. separate vocals
            # TODO: process in memory
            try:
                outputs = self.audio_separator.separate(audio_path)
                if len(outputs) <= 0:
                    raise RuntimeError("Audio separate failed.")

                vocal_audio_file = outputs[0]
                vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
                vocal_audio_file = os.path.join(self.audio_separator.output_dir, vocal_audio_file)
                vocal_audio_file = resample_audio(vocal_audio_file, os.path.join(self.audio_separator.output_dir, f"{vocal_audio_name}-16k.wav"), self.sample_rate)
            except Exception as e:
                log(f"Fail to separate vocals from {audio_path}, error info [{e}]")
                vocal_audio_file=audio_path
        else:
            vocal_audio_file=audio_path
        
        return vocal_audio_file
    
    def load_audio(self, audio_path: str, mono: bool = True, duration: Optional[float] = None) -> Any:
        try:
            audio_data, sampling_rate = librosa.load(audio_path, sr=self.sample_rate, mono=mono, duration=duration)
        except Exception as e:
            raise RuntimeError(f"Fail to load audio from {audio_path}, error info [{e}]")
        return audio_data, sampling_rate

    def prepare_audio_data(self, audio_data: Union[np.ndarray, torch.Tensor], n_frames: Optional[int]=None) -> Tuple[List[Any], int]:
        """Prepare audio data for processing.
        """
        #print(f"==========> Using Wav2Vec2FeatureExtractor to extract audio features")
        audio_data = np.squeeze(self.feature_extractor(audio_data, sampling_rate=self.sample_rate).input_values)

        clip_len = int(len(audio_data) / self.audio_unit)
        if n_frames is not None:
            if abs(n_frames - clip_len) > 7:
                log(f"The number of frames must be close to the clip length (in 280ms), got {n_frames} and {clip_len}")
                return [], n_frames
            clip_len = n_frames
        else:
            n_frames = clip_len

        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data).float().to(self.device)
        assert audio_data.ndim == 1, 'Audio must be 1D tensor.'

        # padding
        # padding audio to fit the clip length
        n_audio_samples = round(self.audio_unit * clip_len)
        n_padding_audio_samples = n_audio_samples - len(audio_data)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0
            elif self.pad_mode == 'replicate':
                padding_value = float(audio_data[-1])
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio_data = F.pad(audio_data, (0, n_padding_audio_samples), value=padding_value)
        
        # devide audio into sub-divisions for saving GPU memory
        audio_segments = []
        if clip_len <= self.subclip_len:
            n_subdivision = 1
            subclip_len = clip_len
        else:
            n_subdivision = math.ceil(clip_len / self.subclip_len)
            subclip_len = self.subclip_len
        
        for i in range(0, n_subdivision):
            start_idx = i * subclip_len
            end_idx = min(start_idx + subclip_len, clip_len)
            # debug
            #log(f"[{i+1}/{n_subdivision}] data index [{round(start_idx * self.audio_unit)}, {round(end_idx * self.audio_unit)})")
            audio_segments.append(
                {
                    "data": audio_data[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "length": end_idx - start_idx
                }
            )
        return audio_segments, n_frames
        
    def get_audio_embedding(self, audio, clip_len: int) -> torch.Tensor:
        if audio.ndim == 2:
            # Extract audio features
            assert audio.shape[1] == 16000 * clip_len / self.fps, \
                f'Incorrect audio length {audio.shape[1]}'
            
            # Extract audio features
            if self.use_half:
                audio = audio.half()
            embeddings = self.audio_encoder(
                pad_audio(audio), seq_len=clip_len, sample_strategy=self.sample_strategy, output_hidden_states=True
            )  # (N, L, 768)
            assert len(embeddings) > 0, "Fail to extract audio embedding"
            
            if self.only_last_features:
                audio_emb = embeddings.last_hidden_state.squeeze(0)
            else:
                audio_emb = torch.stack(
                    embeddings.hidden_states[1:], dim=1
                ).squeeze(0)
                audio_emb = rearrange(audio_emb, "b s d -> s b d")
            
        elif audio.ndim == 3:
            assert audio.shape[1] == clip_len, f'Incorrect audio feature length {audio.shape[1]}'
            audio_emb = audio
        else:
            raise ValueError(f'Incorrect audio input shape {audio.shape}')
        
        return audio_emb

    def get_audio_embeddings(self, audio_segments: List[Any]) -> Optional[torch.Tensor]:
        audio_embs = []
        for audio_segment in audio_segments:
            if self.is_training:
                audio_emb = self.get_audio_embedding(audio_segment["data"], audio_segment["length"])
            else:
                with torch.no_grad():
                    audio_emb = self.get_audio_embedding(audio_segment["data"], audio_segment["length"])
            
            audio_emb = audio_emb.cpu() if self.save_to_cpu else audio_emb
            audio_embs.append(audio_emb)
            #log(f"audio segment [{audio_segment['start_idx']}, {audio_segment['end_idx']}) has been processed.") 
        
        if len(audio_embs) == 0:
            return None

        audio_emb = torch.cat(audio_embs, dim=0)
        
        return audio_emb

    def preprocess(
        self, 
        audio_path: str, 
        n_frames: Optional[int] = None, 
        duration: Optional[float] = None, 
        need_seperate: bool = False
    ):
        """ Preprocess a WAV audio file by separating the vocals from the background and resampling it to a 16 kHz sample rate.
        The separated vocal track is then converted into wav2vec2 for further processing or analysis.
        """
        if need_seperate:
            vocal_audio_file = self.seperate_audio(audio_path)
        else:
            vocal_audio_file = audio_path
        
        audio_data, sampling_rate = self.load_audio(vocal_audio_file, duration=duration)
    
        assert sampling_rate == 16000, "The sample rate of audio must be 16000"
        audio_segments, n_frames = self.prepare_audio_data(audio_data, n_frames)
        audio_emb = self.get_audio_embeddings(audio_segments)
        if audio_emb is None:
            log(f"{audio_path} has been processed, but no audio embedding, set as 'None'.")
        #else:
            #log(f"{audio_path} has been processed, audio embedding shape {audio_emb.shape}.") 
        return audio_emb, n_frames
    
    def preprocess_long(
        self, 
        audio_path: str, 
        need_seperate: bool = False
    ):
        audio_list = cut_audio(audio_path, self.tmp_dir, length=self.max_length)
        audio_emb_list = []
        l = 0

        for idx, audio_path in enumerate(audio_list):
            padding = (idx+1) == len(audio_list)
            emb, length = self.preprocess(audio_path, need_seperate=need_seperate)
            audio_emb_list.append(emb)
            log(f"Processing audio {idx+1}/{len(audio_list)}, path: {audio_path} length: {length}")
            l += length
        
        audio_emb = torch.cat(audio_emb_list)
        audio_length = l

        # remove tmp file
        if len(audio_list) > 1:
            for audio_path in audio_list:
                os.remove(audio_path)
        
        return audio_emb, audio_length

    def add_silent_audio(self, audio_path: str, silent_audio_path: Optional[str] = None, add_duration: float = 1., linear_fusion=False, mode="post"):
        # mode, pre, post, both
        assert mode in ["pre", "post", "both"], f"Unkown mode: {mode}, only support pre, post, both"
        if silent_audio_path is None:
            return audio_path, 0
        else:
            audio_dir = osp.dirname(audio_path)
            audio_name = osp.basename(audio_path)
            temp_audio_path = osp.join(audio_dir, f"tmp_{audio_name}")
            if osp.isfile(temp_audio_path):
                os.remove(temp_audio_path)

            audio, sr1 = librosa.load(audio_path, mono=True, sr=16000)
            # denoise
            audio = librosa.effects.preemphasis(audio)       # enhance voice
            # load silent audio
            silent_audio, sr2 = librosa.load(silent_audio_path, mono=True, sr=16000)
            silent_audio = silent_audio[:int(add_duration*sr2)]
            
            if linear_fusion:
                short_len = min(len(audio), len(silent_audio))
                fusion_ratio = np.linspace(0, 1.0, num=short_len)
                # get pre padding audio
                pre_pad_audio = fusion_ratio * silent_audio[:short_len] + (1 - fusion_ratio) * audio[:short_len]
                if short_len < len(silent_audio):
                    pre_pad_audio = np.hstack((pre_pad_audio, silent_audio[short_len:]))
                pre_pad_audio = np.flip(pre_pad_audio, axis=0)
                
                # get post padding audio
                post_pad_audio = (1 - fusion_ratio) * silent_audio[-short_len:] + fusion_ratio * audio[-short_len:]
                if short_len < len(silent_audio):
                    post_pad_audio = np.hstack((silent_audio[:-short_len], post_pad_audio))
                post_pad_audio = np.flip(post_pad_audio, axis=0)
            else:
                pre_pad_audio = silent_audio
                post_pad_audio = silent_audio
            
            # padding audio
            if mode == "both":
                combined_audio = np.hstack((pre_pad_audio, audio, post_pad_audio))
            elif mode == "pre":
                combined_audio = np.hstack((pre_pad_audio, audio))
            else:
                combined_audio = np.hstack((audio, post_pad_audio))

            add_nframes = math.floor(add_duration * sr2 / self.audio_unit)
            #print(f"audio length: {len(audio)}, pre_pad_audio length: {len(pre_pad_audio)}, post_pad_audio length: {len(post_pad_audio)}, combined_length: {len(combined_audio)}, total add {add_nframes*2} frames")
            #print(f"audio duration: {librosa.get_duration(audio, sr=sr1)}, silent duration: {librosa.get_duration(silent_audio, sr=sr2)}, combined duration: {librosa.get_duration(combined_audio, sr=sr2)}")
            soundfile.write(temp_audio_path, combined_audio, sr2)

            return temp_audio_path, add_nframes
    
    def get_long_audio_emb(self, audio_path: str) -> torch.Tensor:
        audio_emb, length = self.preprocess_long(audio_path)
        log(f"Load audio from {osp.realpath(audio_path)} done, audio_emb shape: {audio_emb.shape}.")
        return audio_emb

    def __enter__(self):
        return self

