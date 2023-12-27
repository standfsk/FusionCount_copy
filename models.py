from torch import Tensor, nn
import torch.nn.functional as F
from helpers import _initialize_weights, ConvNormActivation, FeatureFuser, ChannelReducer
from vgg import VGG
import torch.utils.model_zoo as model_zoo

class FusionCount(nn.Module):
    """
    The official PyTorch implementation of the model proposed in FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion.
    """
    def __init__(self, batch_norm: bool = True) -> None:
        super(FusionCount, self).__init__()
        if batch_norm:
            self.encoder = VGG(name="vgg16_bn", pretrained=True, start_idx=2)
        else:
            self.encoder = VGG(name="vgg16", pretrained=True, start_idx=2)

        self.fuser_1 = FeatureFuser([64, 128, 128], batch_norm=batch_norm)
        self.fuser_2 = FeatureFuser([128, 256, 256, 256], batch_norm=batch_norm)
        self.fuser_3 = FeatureFuser([256, 512, 512, 512], batch_norm=batch_norm)
        self.fuser_4 = FeatureFuser([512, 512, 512, 512], batch_norm=batch_norm)

        self.reducer_1 = ChannelReducer(in_channels=64, out_channels=32, dilation=2, batch_norm=batch_norm)
        self.reducer_2 = ChannelReducer(in_channels=128, out_channels=64, dilation=2, batch_norm=batch_norm)
        self.reducer_3 = ChannelReducer(in_channels=256, out_channels=128, dilation=2, batch_norm=batch_norm)
        self.reducer_4 = ChannelReducer(in_channels=512, out_channels=256, dilation=2, batch_norm=batch_norm)

        output_layer = ConvNormActivation(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            dilation=1,
            norm_layer=None,
            activation_layer=nn.ReLU(inplace=True)
        )

        self.output_layer = _initialize_weights(output_layer)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.encoder(x)

        feat_1, feat_2, feat_3, feat_4 = feats[0: 3], feats[3: 7], feats[7: 11], feats[11:]

        feat_1 = self.fuser_1(feat_1)
        feat_2 = self.fuser_2(feat_2)
        feat_3 = self.fuser_3(feat_3)
        feat_4 = self.fuser_4(feat_4)

        feat_4 = self.reducer_4(feat_4)
        feat_4 = F.interpolate(feat_4, size=feat_3.shape[-2:], mode="bilinear", align_corners=False)

        feat_3 = feat_3 + feat_4
        feat_3 = self.reducer_3(feat_3)
        feat_3 = F.interpolate(feat_3, size=feat_2.shape[-2:], mode="bilinear", align_corners=False)

        feat_2 = feat_2 + feat_3
        feat_2 = self.reducer_2(feat_2)
        feat_2 = F.interpolate(feat_2, size=feat_1.shape[-2:], mode="bilinear", align_corners=False)

        feat_1 = feat_1 + feat_2
        feat_1 = self.reducer_1(feat_1)
        feat_1 = F.interpolate(feat_1, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output = self.output_layer(feat_1)

        return output

model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(args):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls[args.backbone]), strict=False)
    return model