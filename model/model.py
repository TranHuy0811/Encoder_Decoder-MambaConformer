import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder.encoder import Encoder
from .decoder.decoder import Decoder
from .mel_spectrogram import MelSpec
from .spec_augment import SpecAug

class MambaConformer(nn.Module):
    def __init__(self, encoder_dim, vocab_size, padding_idx, bos_idx, eos_idx, # Note: vocab_size without bos and eos tokens
                 sample_rate, window_length, hop_length, nfft, n_mels, mel_scale, # parameters for MelSpectrogram
                 num_freq_masks, freq_mask_range, freq_mask_ratio_range, num_time_masks, time_mask_range, time_mask_ratio_range, # parameters for SpecAugment
                 enc_num_blocks, dec_num_blocks, ff_expand, d_state, d_conv, mamba_expand, conv_expand, depth_ks,
                 dropout_prob=0.1, do_log=True, do_preemp=True, do_spec_aug=False, dec_activation= nn.ReLU
        ):
        super().__init__()
        self.mel_spectrogram = MelSpec(sample_rate, window_length, hop_length, nfft, n_mels, mel_scale, do_log, do_preemp)
        self.spec_aug = (SpecAug(num_freq_masks, freq_mask_range, freq_mask_ratio_range, num_time_masks, time_mask_range, time_mask_ratio_range) if do_spec_aug 
                         else nn.Identity())

        # Encoder
        self.encoder = Encoder(encoder_dim, n_mels,  # in_dim = n_mels
                               enc_num_blocks, ff_expand, d_state, d_conv, mamba_expand, conv_expand, depth_ks, 
                               dropout_prob)
        self.enc_linear = nn.Linear(encoder_dim, vocab_size, bias=False)
        
        # Decoder
        self.decoder = Decoder(encoder_dim, vocab_size, dec_num_blocks, padding_idx, d_state, d_conv, mamba_expand, ff_expand, dec_activation, dropout_prob)
        self.dec_linear = nn.Linear(encoder_dim, vocab_size + 2, bias=False) # vocab_size along with bos and eos tokens
        


    def forward(self, src, tgt, train=False): # src, tgt: (batch, token_len)
        with torch.no_grad():
            src = self.mel_spectrogram(src) # (batch, token_len) => (batch, freq, time)
            if train:
                src = self.spec_aug(src)


        # Encoder Loss (CTC loss)
        src_out = self.encoder(src) # (batch, freq, time) => (batch, time, encoder_dim)
        src_pred = self.enc_linear(src_out)
        src_pred = F.log_softmax(src_pred, dim=-1)


        # Decoder Loss (Attention-based loss)
        tgt_out = self.decoder(tgt, src_out)
        tgt_pred = self.dec_linear(tgt_out)


        return src_pred, tgt_pred