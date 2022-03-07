import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.extend("../../../")

from ...log.Log import Log

class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]
        self.client_number = args["client_number"]
        self.state = "A"
        self.total_loss = 0.0
        self.last_update_loss = 0.0
        self.delta_loss = 0.0
        self.total_batch = 0
        self.current_batch = 0
        self.loss_thred = 0.030
        self.acts_list = dict()
        self.label_list = dict()
        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.active_node = 1
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts, labels):
        if self.state == 'C':
            self.acts = self.acts_list[self.current_batch]
            acts = self.acts_list[self.current_batch]
            labels = self.label_list[self.current_batch]

        else:
            self.acts = acts
            self.acts.retain_grad()
            self.acts_list[self.current_batch] = self.acts
            self.label_list[self.current_batch] = labels
        self.current_batch += 1
        self.optimizer.zero_grad()
        self.acts.retain_grad()

        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total_loss += self.loss
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        if self.step % self.log_step == 0 and self.phase == "train":
            acc = self.correct / self.total
            self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format("train", acc, self.loss.item(), self.epoch, self.step))

            # 用log记录一下准确率之类的信息
        if self.phase == "validation":
            # self.log.info("phase={} acc={} loss={} epoch={} and step={}"
            #               .format("train", acc, self.loss.item(), self.epoch, self.step))
            self.val_loss += self.loss.item()
            # torch.save(self.model, self.args["model_save_path"].format("server", self.epoch, ""))
        self.step += 1

    def backward_pass(self):
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        if self.state == 'A':
            return self.acts.grad
        else:
            return None

    def validation_over(self):
        # not precise estimation of validation loss
        self.val_loss /= self.step
        acc = self.correct / self.total

        # 这里也要用log记录一下准确率之类的信息
        self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format(self.phase, acc, self.val_loss, self.epoch, self.step))
        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()

    def update_state(self):
        self.current_batch = 0
        if self.state == "A":
            self.last_update_loss = self.total_loss / self.total_batch
        self.delta_loss = self.last_update_loss - self.total_loss / self.total_batch
        if self.delta_loss >= self.loss_thred:
            self.state = "A"
        else:
            if self.state == 'A':
                self.state = 'B'
            else:
                self.state = 'C'
        self.log.info("state: {} last Loss: {} delta_loss: {}".format(self.state, self.last_update_loss,
                                                                     self.delta_loss))
        return self.state


