import torch

import math
import os
import kaldiio
import kaldi_native_fbank as knf
import numpy as np


from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor
from nemo.core.classes import Exportable

class FireRedPreprocessor(AudioPreprocessor, Exportable):
    """
        borrowed from FireRedASR/examples/fireredasr/data/asr_feat.py
    """
    def __init__(self, kaldi_cmvn_file):
        super().__init__(win_length=0.025, hop_length=0.01)
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
            frame_shift=10, dither=0.0)

        self._sample_rate = 16000

    # def __call__(self, wav_paths):
    #     feats = []
    #     durs = []
    #     for wav_path in wav_paths:
    #         sample_rate, wav_np = kaldiio.load_mat(wav_path)
    #         dur = wav_np.shape[0] / sample_rate
    #         fbank = self.fbank((sample_rate, wav_np))
    #         if self.cmvn is not None:
    #             fbank = self.cmvn(fbank)
    #         fbank = torch.from_numpy(fbank).float()
    #         feats.append(fbank)
    #         durs.append(dur)
    #     lengths = torch.tensor([feat.size(0) for feat in feats]).long()
    #     feats_pad = self.pad_feat(feats, 0.0)
    #     return feats_pad, lengths, durs

    def pad_feat(self, xs, pad_value):
        # type: (List[Tensor], int) -> Tensor
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad

    def get_features(self, input_signal, length):
        device = input_signal.device
        fbanks, lengths = [], []

        for i in range(input_signal.shape[0]):
            fbank = self.fbank((16000, input_signal[i].cpu().numpy()))
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            fbank = torch.from_numpy(fbank).float().to(device)
            fbanks.append(fbank)
            lengths.append(fbank.shape[0])
        fbanks = torch.stack(fbanks, dim=0)
        return fbanks, torch.tensor(lengths)

class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean*mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)



class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav

        if len(wav_np.shape) != 1:
            wav_np = wav_np[0]


        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat
