from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers import HubertModel
from transformers.modeling_outputs import BaseModelOutput


_CONFIG_FOR_DOC = 'HubertConfig'


def linear_interpolation(features, seq_len):
    """
    Transpose the features to interpolate linearly.

    Args:
        features (torch.Tensor): The extracted features to be interpolated.
        seq_len (torch.Tensor): The sequence lengths of the features.

    Returns:
        torch.Tensor: The interpolated features.
    """
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class HubertModel_(HubertModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        seq_len: Optional[int],
        sample_strategy: Optional[str] = "presample",
        attention_mask: Optional[torch.LongTensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the HuBERT model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            sample_strategy: The sample strategy to align features and seq_len, supports ['presample', 'postsample'].
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the HuBERT model.
        """
        # output_fps=25, 
        # attention_mask=None, 
        # output_attentions=None,
        # output_hidden_states=None, 
        # return_dict=None, 
        # frame_num=None
        assert sample_strategy in ["presample", "postsample"], f"sample_strategy must be in ['presample', 'postsample]"
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)  # (N, C, L)
        extract_features = extract_features.transpose(1, 2)
        if sample_strategy == "presample":  
            extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        # # Resample the audio feature @ 50 fps to `output_fps`.
        # if frame_num is not None:
        #     extract_features_len = round(frame_num * 50 / output_fps)
        #     extract_features = extract_features[:, :, :extract_features_len]
        # extract_features = linear_interpolation(extract_features, 50, output_fps, output_len=frame_num)
        # extract_features = extract_features.transpose(1, 2)  # (N, L, C)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, 
            mask_time_indices=mask_time_indices, 
            attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if sample_strategy == "postsample":
            hidden_states = linear_interpolation(hidden_states, seq_len=seq_len)
            for i in range(len(encoder_outputs.hidden_states)):
                encoder_outputs.hidden_states[i] = linear_interpolation(encoder_outputs.hidden_states[i], seq_len=seq_len)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states, 
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions, 
        )
