import argparse
import torch

from PIL import Image
from torchvision.transforms import functional as F
from models import StyleTransfer, load_model


def from_image(image):
    x = F.to_tensor(image).unsqueeze(0)

    if torch.cuda.is_available():
        x = x.cuda()

    return x


def to_image(x):
    return F.to_pil_image(x.squeeze(0).cpu())


def style_transfer(model, source, size):
    image = Image.open(source).convert(mode='RGB')

    if size[0] and size[1]:
        image = image.resize(size)

    x = from_image(image)
    x = model(x)

    return to_image(x)


def main():
    parser = argparse.ArgumentParser(description='StyleTransfer Single Test')

    parser.add_argument('input', type=str,
                        help='Input image path')
    parser.add_argument('--width', type=int, default=None,
                        help='Width of output image')
    parser.add_argument('--height', type=int, default=None,
                        help='Height of output image')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--output', default=None,
                        help='Save filename')
    args = parser.parse_args()

    # build model
    model = StyleTransfer()

    # load weight
    if args.weight:
        weight = args.weight
    else:
        weight = 'checkpoints/style_transfer_latest.pth'

    load_model(model, weight)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # inference image
    with torch.no_grad():
        image = style_transfer(model, args.input, (args.width, args.height))

    # show style transfered image
    if args.output:
        image.save(args.output)
    else:
        image.show()


if __name__ == "__main__":
    main()