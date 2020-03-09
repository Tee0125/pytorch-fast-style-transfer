import torch

from torch import nn
from torch.functional import F
from torchvision.models import vgg16, vgg19


class StyleTransferLoss(nn.Module):
    def __init__(self, style_image, args):
        super().__init__()

        backbone = self.init_backbone(args)

        self.backbone_content = backbone[0]
        self.backbone_style = backbone[1]

        self.w_content = args.lambda_content
        self.w_style = args.lambda_style
        self.w_tv = args.lambda_tv

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        if torch.cuda.is_available():
            mean = mean.cuda()
            std = std.cuda()

        self.mean = mean
        self.std = std

        if torch.cuda.is_available():
            style_image = style_image.cuda()

        style_image = self.normalize_image(style_image.unsqueeze(0))

        self.style_features = self.get_style_features(style_image)

    def __call__(self, y_, y):
        if torch.cuda.is_available():
            y = y.cuda()

        y = self.normalize_image(y)
        y_ = self.normalize_image(y_)

        loss = []

        loss.append(self.w_content * self.get_content_loss(y_, y))
        loss.append(self.w_style * self.get_style_loss(y_))
        loss.append(self.w_tv * self.get_tv_loss(y_))

        return sum(loss)

    def get_content_loss(self, y_, y):
        y_ = self.backbone_content(y_)
        y = self.backbone_content(y)

        return F.mse_loss(y_, y)

    def get_style_loss(self, y_):
        n = y_.size(0)

        y_ = self.get_style_features(y_)
        y = self.style_features

        l_style = 0

        for g_, g in zip(y_, y):
            l_style += F.mse_loss(g_, g.expand_as(g_), reduction='sum')

        return l_style / n

    @staticmethod
    def get_tv_loss(y_):
        dx = F.mse_loss(y_[:, :, :, 1:], y_[:, :, :, :-1])
        dy = F.mse_loss(y_[:, :, 1:, :], y_[:, :, :-1, :])

        return dx + dy

    def get_style_features(self, x):
        features = []

        for layer in self.backbone_style:
            x = layer(x)
            features.append(self.gram_matrix(x))

        return features

    def normalize_image(self, x):
        return (x - self.mean) / self.std

    @staticmethod
    def init_backbone(args):
        if args.use_vgg19:
            vgg = vgg19(pretrained=True)
            features = vgg.features[0:30]
        else:
            vgg = vgg16(pretrained=True)
            features = vgg.features[0:23]

        # freeze parameters
        features.eval()

        for p in features.parameters():
            p.requires_grad = False

        if torch.cuda.is_available():
            features = features.cuda()

        if args.use_vgg19:
            # relu4_2
            backbone_content = features[0:23]

            # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
            backbone_style = [features[0:2],
                              features[2:7],
                              features[7:12],
                              features[12:21],
                              features[21:30]]
        else:
            # relu2_2
            backbone_content = features[0:9]

            # relu1_2, relu2_2, relu3_3, relu4_3
            backbone_style = [features[0:4],
                              features[4:9],
                              features[9:16],
                              features[16:23]]

        backbone_style = nn.ModuleList(backbone_style)

        return backbone_content, backbone_style

    @staticmethod
    def gram_matrix(x):
        n = x.numel() / x.size(0)

        x = x.reshape(x.size(0), x.size(1), -1)
        x = torch.bmm(x, x.transpose(1, 2))

        return x / n

