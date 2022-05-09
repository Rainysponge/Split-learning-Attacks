import torch.optim as optim
import logging
from ...log.Log import Log
import torch.nn as nn

class SplitNNClient():

    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"][0]
        self.model_2 = args["client_model"][1]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.SERVER_RANK = args["server_rank"]

        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.local_sample_number = len(self.trainloader)
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9, weight_decay=5e-4)
        self.optimizer_2 = optim.SGD(self.model_2.parameters(), args["lr"], momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.device = args["device"]

        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = 3
        self.MAX_EPOCH_PER_NODE = args["epochs"]

        self.phase = "train"

        self.log = Log(self.__class__.__name__, args)
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.args = args

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def write_log(self):
        if (self.phase == "train" and self.step%self.log_step==0) or self.phase=="validation":
            self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format(self.phase, self.correct/self.total, self.val_loss, self.epoch_count, self.step))

    def forward_pass(self, type=0, inputs=None):
        # logging.info("{} begin run_forward_pass".format(self.rank))
        # inputs, labels = next(self.dataloader)
        #
        # inputs, labels = inputs.to(self.device), labels.to(self.device)
        # logging.info("img size:{}".format(inputs.shape))
        # self.optimizer.zero_grad()
        #
        # self.acts = self.model(inputs)
        # return self.acts, labels
        if type == 0:
            inputs, labels = next(self.dataloader)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.labels = labels
            self.optimizer.zero_grad()
            self.acts = self.model(inputs)
            return self.acts
        else:
            self.optimizer_2.zero_grad()
            self.acts2 = inputs
            self.acts2.retain_grad()
            logits = self.model_2(self.acts2)
            _, predictions = logits.max(1)
            self.loss = self.criterion(logits, self.labels)

            self.total += self.labels.size(0)
            self.correct += predictions.eq(self.labels).sum().item()
            if self.step % self.log_step == 0 and self.phase == "train":
                acc = self.correct / self.total
                self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                              .format("train", acc, self.loss.item(), self.epoch_count, self.step))
                # 用log记录一下准确率之类的信息
            if self.phase == "validation":
                self.val_loss += self.loss.item()
                # torch.save(self.model, self.args["model_save_path"].format("server", self.epoch_count, ""))
            self.step += 1

    def backward_pass(self, type=0, grads=None):
        # self.acts.backward(grads)
        # self.optimizer.step()
        if type == 0:
            self.acts.backward(grads)
            self.optimizer.step()
        else:
            self.loss.backward(retain_graph=True)
            self.optimizer_2.step()
            return self.acts2.grad

    """
    如果模型有dropout或者BN层的时候，需要改变模型的模式
    """

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.phase="validation"
        self.model.eval()
        self.reset_local_params()


    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.phase="train"
        self.model.train()
        self.reset_local_params()
