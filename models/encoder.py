import torch
import torch.nn as nn
from .blocks import(
    ConvBlock,
    OctaveConv,
    DSCBlock
)


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        
        input_channels = 3

        vanilla_config = model_config['vanilla']
        octave_config = model_config['octave']
        dsc_config = model_config['dsc']
        
        for i in range(vanilla_config['num_layers']):
            self.layers.append(
                ConvBlock(input_channels, vanilla_config['num_filters'], vanilla_config['kernel_size'])
                )
            input_channels = vanilla_config['num_filters']

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
