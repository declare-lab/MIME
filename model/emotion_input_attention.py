### MOST OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
## MINOR CHANGES
# import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
import os
from utils import config
from utils.metric import rouge, moses_multi_bleu, _prec_recall_f1_score, compute_prf, compute_exact_match

# if(config.model == 'multi-trs'):
#     from utils.beam_omt_multiplex import Translator
# else:
#     from utils.beam_omt import Translator
if (config.model == 'trs'):
    from utils.beam_omt import Translator
elif (config.model == 'seq2seq'):
    from utils.beam_ptr import Translator
elif (config.model == 'multi-trs'):
    from utils.beam_omt_multiplex import Translator
elif (config.model == 'experts'):
    from utils.beam_omt_experts import Translator
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=1)
import numpy as np
from model.transformer_mulexpert import Encoder
from model.transformer_mulexpert import Decoder

from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, \
    get_input_from_batch, get_output_from_batch, top_k_top_p_filtering, evaluate, count_parameters, make_infinite

# import matplotlib.pyplot as plt


class EmotionInputEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layers, num_heads,
                 total_key_depth, total_value_depth,
                 filter_size, universal, emo_input):

        super(EmotionInputEncoder, self).__init__()
        self.emo_input = emo_input
        if self.emo_input == "self_att":
            self.enc = Encoder(2 * emb_dim, hidden_size, num_layers, num_heads,
                               total_key_depth, total_value_depth,
                               filter_size, universal=universal)
        elif self.emo_input == "cross_att":
            self.enc = Decoder(emb_dim, hidden_size, num_layers, num_heads,
                               total_key_depth, total_value_depth,
                               filter_size, universal=universal)
        else:
            raise ValueError("Invalid attention mode.")

    def forward(self, emotion, encoder_outputs, mask_src):
        if self.emo_input == "self_att":
            repeat_vals = [-1] + [encoder_outputs.shape[1] // emotion.shape[1]] + [-1]
            hidden_state_with_emo = torch.cat([encoder_outputs, emotion.expand(repeat_vals)], dim=2)
            return self.enc(hidden_state_with_emo, mask_src)
        elif self.emo_input == "cross_att":
            return self.enc(encoder_outputs, emotion, (None, mask_src))[0]
