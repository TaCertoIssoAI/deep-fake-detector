"""
AI-Synthesized Voice Generalization detector model.

Reimplemented from: https://github.com/Purdue-M2/AI-Synthesized-Voice-Generalization
Paper: "Improving Generalization for AI-Synthesized Voice Detection" (AAAI 2025)

Architecture:
- Dual RawNet2 encoders (forgery + content) with SincConv frontend
- Domain-specific and domain-agnostic feature splitting via Conv2d 1x1
- Classification heads for specific (7-class vocoder) and shared (binary) tasks
- Only the shared head is used at inference
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SincConv — learnable sinc-based bandpass filterbank
# ---------------------------------------------------------------------------


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        device,
        out_channels,
        kernel_size,
        in_channels=1,
        sample_rate=16000,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        freq_scale="Mel",
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel")

        self.out_channels = out_channels + 1
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.sample_rate = sample_rate
        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)

        if freq_scale == "Mel":
            fmel = self.to_mel(f)
            filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 2)
            filbandwidthsf = self.to_hz(filbandwidthsmel)
            self.freq = filbandwidthsf[: self.out_channels]
        elif freq_scale == "Inverse-mel":
            fmel = self.to_mel(f)
            filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 2)
            filbandwidthsf = self.to_hz(filbandwidthsmel)
            self.freq = np.abs(np.flip(filbandwidthsf[: self.out_channels]) - 1)
        else:
            filbandwidthsmel = np.linspace(np.min(f), np.max(f), self.out_channels + 2)
            self.freq = filbandwidthsmel[: self.out_channels]

        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels - 1, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.freq) - 1):
            fmin = self.freq[i]
            fmax = self.freq[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = torch.tensor(np.hamming(self.kernel_size)) * torch.tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)
        self.filters = band_pass_filter.view(self.out_channels - 1, 1, self.kernel_size)
        return F.conv1d(x, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], kernel_size=3, padding=1, stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=1, padding=0, stride=1)
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        out = self.mp(out)
        return out


# ---------------------------------------------------------------------------
# RawNet2 backbone
# ---------------------------------------------------------------------------


class RawNet(nn.Module):
    """RawNet2 encoder backbone. Returns 1024-dim feature vector (before final FC)."""

    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        self.device = config["device"]

        self.Sinc_conv = SincConv(
            device=self.device,
            out_channels=config["filts"][0],
            kernel_size=config["first_conv"],
            in_channels=config["in_channels"],
            freq_scale="Mel",
        )

        self.first_bn = nn.BatchNorm1d(num_features=config["filts"][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts=config["filts"][1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts=config["filts"][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts=config["filts"][2]))
        config["filts"][2][0] = config["filts"][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=config["filts"][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts=config["filts"][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts=config["filts"][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = nn.Sequential(nn.Linear(config["filts"][1][-1], config["filts"][1][-1]))
        self.fc_attention1 = nn.Sequential(nn.Linear(config["filts"][1][-1], config["filts"][1][-1]))
        self.fc_attention2 = nn.Sequential(nn.Linear(config["filts"][2][-1], config["filts"][2][-1]))
        self.fc_attention3 = nn.Sequential(nn.Linear(config["filts"][2][-1], config["filts"][2][-1]))
        self.fc_attention4 = nn.Sequential(nn.Linear(config["filts"][2][-1], config["filts"][2][-1]))
        self.fc_attention5 = nn.Sequential(nn.Linear(config["filts"][2][-1], config["filts"][2][-1]))

        self.bn_before_gru = nn.BatchNorm1d(num_features=config["filts"][2][-1])
        self.gru = nn.GRU(
            input_size=config["filts"][2][-1],
            hidden_size=config["gru_node"],
            num_layers=config["nb_gru_layer"],
            batch_first=True,
        )

        self.fc1_gru = nn.Linear(config["gru_node"], config["nb_fc_node"])
        self.fc2_gru = nn.Linear(config["nb_fc_node"], config["nb_classes"], bias=True)
        self.sig = nn.Sigmoid()

    def features(self, x, inference=False):
        """Extract 1024-dim features (before final classification FC)."""
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        for block, attn in [
            (self.block0, self.fc_attention0),
            (self.block1, self.fc_attention1),
            (self.block2, self.fc_attention2),
            (self.block3, self.fc_attention3),
            (self.block4, self.fc_attention4),
            (self.block5, self.fc_attention5),
        ]:
            x_out = block(x)
            y = self.avgpool(x_out).view(x_out.size(0), -1)
            y = attn(y)
            y = self.sig(y).view(y.size(0), y.size(1), -1)
            x = x_out * y + y

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        return x

    def forward(self, x, inference=False):
        x = self.features(x, inference)
        x = self.fc2_gru(x)
        if inference:
            return F.softmax(x, dim=1)
        return x


# ---------------------------------------------------------------------------
# Helper modules for the AudioFakeDetector
# ---------------------------------------------------------------------------


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )

    def forward(self, x):
        bs = x.size(0)
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


# ---------------------------------------------------------------------------
# Full AudioFakeDetector (inference-only)
# ---------------------------------------------------------------------------

RAWNET_CONFIG = {
    "device": "cpu",
    "nb_samp": 64000,
    "first_conv": 128,
    "in_channels": 1,
    "filts": [128, [128, 128], [128, 512], [512, 512]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
}


class AudioFakeDetector(nn.Module):
    """
    Dual-encoder disentanglement detector for AI-synthesized voice detection.

    At inference, only the shared (domain-agnostic) classification head is used.
    The forgery encoder extracts features, which are split into domain-specific
    and domain-agnostic components. The shared head outputs [real, fake] logits.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        config = copy.deepcopy(RAWNET_CONFIG)
        config["device"] = device
        self.encoder_feat_dim = 1024
        self.half_fingerprint_dim = self.encoder_feat_dim

        # Dual encoders
        self.encoder_f = RawNet(config)  # forgery encoder
        self.encoder_c = RawNet(config)  # content encoder

        # Feature splitting blocks
        self.block_spe = Conv2d1x1(self.encoder_feat_dim, self.half_fingerprint_dim, self.half_fingerprint_dim)
        self.block_sha = Conv2d1x1(self.encoder_feat_dim, self.half_fingerprint_dim, self.half_fingerprint_dim)

        # Classification heads
        specific_task_number = 7  # 7 vocoder classes for LibriSeVoc
        self.head_spe = Head(self.half_fingerprint_dim, self.encoder_feat_dim, specific_task_number)
        self.head_sha = Head(self.half_fingerprint_dim, self.encoder_feat_dim, 2)  # binary: real/fake

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward pass.

        Args:
            audio: raw waveform tensor [batch, samples] (64000 samples at 16kHz)

        Returns:
            logits: [batch, 2] — class 0 = real, class 1 = fake
        """
        # Extract forgery features
        forgery_features = self.encoder_f.features(audio)

        # Split into specific and shared
        f = forgery_features.unsqueeze(2).unsqueeze(3)
        f_share = self.block_sha(f)

        # Classify using shared (domain-agnostic) head
        out_sha, _ = self.head_sha(f_share)
        return out_sha
