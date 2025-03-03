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
        ##

        vanilla_config = model_config['Vanilla']
        octave_config = model_config['Octave']
        dsc_config = model_config['Separable_DepthWise_Block']
        
        self.layers = nn.ModuleList()

        in_channels = vanilla_config["in_channels"]

        self.layers.append(
            ConvBlock(
                in_=in_channels,
                out_=vanilla_config["out_channels"],
                stride=vanilla_config["stride"],
                k=vanilla_config["k"]
            )
        )

        in_channels = vanilla_config["out_channels"]

        ## octave conv
        for i in range(octave_config["num_layers"]):
            self.layers.append(
                OctaveConv(
                    in_channels=in_channels,
                    out_channels=octave_config["out_channels"][i],
                    kernel_size=3,
                    alpha_in=octave_config["octave_alpha"][i],
                    alpha_out=octave_config["octave_alpha"][i],
                    stride=octave_config["stride"][i],
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=False
                )
            )
            in_channels = octave_config["out_channels"][i]
            if octave_config["max_pool"][i] != 0:
                self.layers.append(
                    nn.MaxPool2d(
                        kernel_size=octave_config["max_pool"][i],
                        stride=octave_config["max_pool"][i]
                    )
                )

        ## dsc conv
        self.layers.append(
            DSCBlock(
                in_=in_channels,
                out_=dsc_config["out_channels"],
                dropout=0.4
            )
        )

    def forward(self, x):
        print (x.size())
        for layer in self.layers:
            # print (x.size())
            x = layer(x)
        return x


if __name__ == "__main__":
    ## load yaml config
    import yaml
    with open("./config/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    model = FCN_Encoder(model_config['model_config'])
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())
