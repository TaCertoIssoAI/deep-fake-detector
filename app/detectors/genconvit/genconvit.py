import torch
import torch.nn as nn

from app.detectors.genconvit.genconvit_ed import GenConViTED
from app.detectors.genconvit.genconvit_vae import GenConViTVAE


class GenConViT(nn.Module):
    """Combined GenConViT model running both ED and VAE networks."""

    def __init__(self, config):
        super().__init__()
        self.ed = GenConViTED(config, pretrained=False)
        self.vae = GenConViTVAE(config, pretrained=False)

    def forward(self, x):
        ed_out = self.ed(x)
        vae_out, _ = self.vae(x)
        return torch.cat((ed_out, vae_out), dim=0)
