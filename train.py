import argparse

from trainers import StyleTransferTrainer
from utils import TrainerCallback


class Callback(TrainerCallback):
    def __init__(self, args):
        super().__init__()

        # print status
        print("Start Training")
        print("")

        print("Batch Size: %d" % args.batch_size)
        print("Learning Rate: %.3f" % args.lr)
        print("")

        print("lambda style: %.1f" % args.lambda_style)
        print("lambda content: %.1f" % args.lambda_content)
        print("lambda total variance: %.1f" % args.lambda_tv)
        print("")

        self.min_loss = 99.99
        self.minibatch_cnt = 0

    def step_start(self, t, epoch):
        self.minibatch_cnt = 0

    def minibatch_end(self, t, epoch, idx, loss):
        print("epoch#%d, minibatch#%d - loss: %.3f" % (epoch, idx, loss))

        self.minibatch_cnt += 1

        if (self.minibatch_cnt % 500) == 0:
            postfix = str(self.minibatch_cnt // 500)
            t.save_model(postfix=postfix)

    def fit_end(self, t):
        t.save_model(postfix='latest')


def main():
    parser = argparse.ArgumentParser(description='StyleTransfer Training')

    parser.add_argument('--style_image',
                        default='samples/style/rain_princess.jpg',
                        help='Style image path')
    parser.add_argument('--dataset_root',
                        default='downloads',
                        help='Dataset root directory path')
    parser.add_argument('--width', type=int,
                        default=256,
                        help='Output width')
    parser.add_argument('--height', type=int,
                        default=256,
                        help='Output height')
    parser.add_argument('--batch_size', type=int,
                        default=4,
                        help='Batch size for training')
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', type=int,
                        default=-1,
                        help='Number of workers used in dataloading')
    parser.add_argument('--epochs', type=int,
                        default=2,
                        help='Number of epochs to run')
    parser.add_argument('--lambda_style', type=float,
                        default=1.5e1,
                        help='Weight for style loss')
    parser.add_argument('--lambda_content', type=float,
                        default=7.5e0,
                        help='Weight for feature loss')
    parser.add_argument('--lambda_tv', type=float,
                        default=2.0e2,
                        help='Weight for total variance loss')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='Initial learning rate')
    parser.add_argument('--use_vgg19', action='store_true',
                        default=False,
                        help='Use VGG19 to calculate perceptual loss')
    args = parser.parse_args()

    t = StyleTransferTrainer(args, callback=Callback(args))
    t.fit()


if __name__ == "__main__":
    main()

