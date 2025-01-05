import torchaudio
import torch
import torch.nn.functional as F
import torch.nn as nn


class MelSpec(nn.Module):
    def __init__(self, sample_rate, window_length, hop_length, nfft, n_mels, mel_scale, do_log=True, do_preemp=True):
        super().__init__()
        self.do_log = do_log
        self.do_preemp = do_preemp
        self.transform = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sample_rate, n_fft=nfft,
                            win_length=window_length, hop_length=hop_length,
                            n_mels=n_mels, mel_scale=mel_scale,
                            window_fn=torch.hamming_window)
        
        if self.do_preemp:
            self.register_buffer(
                "flipped_filter",
                torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0),
            )



    def forward(self, x):
        if self.do_preemp:
            # reflect padding to match lengths of in/out
            x = x.unsqueeze(1)
            x = F.pad(x, (1, 0), "reflect")
            # apply preemphasis
            x = F.conv1d(x, self.flipped_filter).squeeze(1)

        x = self.transform(x)

        if self.do_log:
            x = torch.log(x + 1e-6)
        
        return x