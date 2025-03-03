import torch
import torch.nn as nn
from .blocks import(
    ConvBlock,
    OctaveConv,
    DSCBlock
)

class FCN_Encoder(nn.Module):
    def __init__(self, model_config):
        super(FCN_Encoder, self).__init__()

        ##
        import yaml
        with open("./config/model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)
        model_config = model_config['model_config']
        ##
        
        vanilla_config = model_config['Vanilla']
        octave_config = model_config['Octave']
        dsc_config = model_config['Separable_DepthWise_Block']
        
        self.vanilla_layers = nn.ModuleList()

        in_channels = vanilla_config["in_channels"]

        self.vanilla_layers.append(
            ConvBlock(
                in_=in_channels,
                out_=vanilla_config["out_channels"],
                stride=vanilla_config["stride"],
                k=vanilla_config["k"]
            )
        )

        in_channels = vanilla_config["out_channels"]

        self.octave_layers = nn.ModuleList()
        ## octave conv
        for i in range(octave_config["num_layers"]):
            self.octave_layers.append(
                OctaveConv(
                    in_channels=in_channels,
                    out_channels=octave_config["out_channels"][i],
                    kernel_size=octave_config["kernel_size"][i],
                    alpha_in=0 if i == 0 else octave_config["octave_alpha"][i-1],
                    alpha_out=octave_config["octave_alpha"][i],
                    stride=octave_config["stride"][i],
                    padding=octave_config["padding"][i],
                    dilation=1,
                    groups=1,
                    bias=False
                )
            )
            in_channels = octave_config["out_channels"][i]
            if octave_config["max_pool"][i] != 0:
                self.octave_layers.append(
                    nn.MaxPool2d(
                        kernel_size=octave_config["max_pool"][i],
                        stride=octave_config["max_pool"][i]
                    )
                )

        ## dsc conv
        self.dsc_layers = nn.ModuleList()
        self.dsc_layers.append(
            DSCBlock(
                in_=in_channels,
                out_=dsc_config["out_channels"],
                dropout=0.4
            )
        )

    def forward(self, x):
        ## vanilla conv
        for layer in self.vanilla_layers:
            # print(f'van, {x.size()}')
            x = layer(x)
        ## octave conv
        for layer in self.octave_layers:
            # if isinstance(x, tuple):
                # print(f'oct, {x[0].size()}', x[1].size())
            x = layer(x)

        # assert x[1] is None
        x = x[0]

        ## dsc conv
        for layer in self.dsc_layers:
            # print(f'dsc, {x.size()}')
            x = layer(x)
        return x


if __name__ == "__main__":
    ## load yaml config
    import yaml
    with open("./config/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    model = FCN_Encoder(model_config['model_config'])
    x = torch.randn(16, 3, 90, 1185)
    y = model(x)
    print(y.size())
