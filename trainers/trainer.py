import os
import multiprocessing
import torch

from models import load_model, save_model


class Trainer:
    def __init__(self, args, callback=None):
        self.args = args

        self.model = self.init_model()
        self.load_model()

        self.dataloader = self.init_dataloader()

        self.criterion = self.init_loss()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

        self.callback = callback

    def fit(self):
        if self.callback:
            self.callback.fit_start(self)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            loss = self.step(epoch)

        if self.callback:
            self.callback.fit_end(self)

    def step(self, epoch):
        if self.callback:
            self.callback.step_start(self, epoch)

        losses = []

        self.model.train()
        for i, batch in enumerate(self.dataloader):
            loss = self.minibatch(epoch, i, batch)
            losses.append(loss)

        loss = sum(losses) / len(losses)

        if self.scheduler:
            self.scheduler.step()

        if self.callback:
            self.callback.step_end(self, epoch, loss)

        return loss
        
    def minibatch(self, epoch, idx, batch):
        if self.callback:
            self.callback.minibatch_start(self, epoch, idx)

        x, y = batch

        if torch.cuda.is_available():
            x = x.cuda()

        y_ = self.model.forward(x)

        loss = self.criterion(y_, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.item()

        if self.callback:
            self.callback.minibatch_end(self, epoch, idx, loss)

        return loss

    def init_dataloader(self):
        raise Exception('Not implemented yet')
    
    def init_model(self):
        raise Exception('Not implemented yet')

    def init_loss(self):
        raise Exception('Not implemented yet')

    def init_optimizer(self):
        raise Exception('Not implemented yet')

    def init_scheduler(self):
        raise Exception('Not implemented yet')

    def get_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        return model

    def load_model(self):
        if self.args.resume:
            load_model(self.get_model(), self.args.resume)

    def save_model(self, path='./checkpoints', postfix=None):
        save_model(self.get_model(), path, postfix)

