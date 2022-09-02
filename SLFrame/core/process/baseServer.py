import torch.nn as nn
import torch.optim as optim
import logging
from core.log.Log import Log


class baseServer():
    def __init__(self, args):

        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.validation_sign_number = 0

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train_mode(self):
        self.model.train()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.reset_local_params()
        self.phase = "validation"

    def forward_pass(self, acts, labels):
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()

        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total = labels.size(0)
        self.correct = predictions.eq(labels).sum().item()
        self.val_loss = self.loss.item()

    def backward_pass(self):
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        # self.log.info(self.acts.grad.shape)
        return self.acts.grad

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def optimizerFactory(self, args):
        # optimizer
        try:
            if args['optimizer'] == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                           weight_decay=5e-4)
            elif args['optimizer'] == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=args["lr"],
                                            betas=(0.9, 0.999),
                                            eps=1e-08,
                                            weight_decay=0,
                                            amsgrad=False)

        except:
            pass
