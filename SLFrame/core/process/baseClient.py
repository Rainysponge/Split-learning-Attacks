import torch.optim as optim
import logging
from core.log.Log import Log


class baseClient():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.rank = args["rank"]

        self.log = Log(self.__class__.__name__, args)
        self.MAX_RANK = args["max_rank"]
        self.device = args["device"]
        self.phase = "train"
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = 3
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.optimizer = None
        self.log = Log(self.__class__.__name__, args)
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.args = args

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

    def forward_pass(self):
        inputs, labels = next(self.dataloader)

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        self.acts = self.model(inputs)
        logging.info("{} forward_pass".format(self.rank))
        return self.acts, labels

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.phase = "validation"
        self.model.eval()
        self.reset_local_params()

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.phase = "train"
        self.model.train()
        self.reset_local_params()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0
