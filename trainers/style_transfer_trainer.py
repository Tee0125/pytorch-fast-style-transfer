import os
import multiprocessing
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import StyleTransfer
from losses import StyleTransferLoss

from .trainer import Trainer


class StyleTransferTrainer(Trainer):
    def __init__(self, args, callback=None):
        super().__init__(args, callback)

    def init_dataloader(self):
        args = self.args

        size = (args.width, args.height)
        root = os.path.join(args.dataset_root)

        t = transforms.Compose((transforms.Resize(size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()))

        dataset = ImageFolder(root, transform=t)

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers

        return DataLoader(dataset,
                          shuffle=True,
                          batch_size=args.batch_size,
                          num_workers=num_workers,
                          collate_fn=self.collate)

    def init_model(self):
        args = self.args

        model = StyleTransfer()

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        return model

    def init_loss(self):
        args = self.args

        size = (args.width, args.height)

        t = transforms.Compose((transforms.Resize(size),
                                transforms.ToTensor()))

        image = Image.open(args.style_image).convert(mode='RGB')
        loss = StyleTransferLoss(t(image), args)

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def init_optimizer(self):
        args = self.args

        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=args.lr)

        return optimizer

    def init_scheduler(self):
        return None

    def collate(self, batch):
        features = [feature for feature, _ in batch]

        target_feature = torch.stack(features, 0)

        return target_feature, target_feature

