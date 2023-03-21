import logging
import torch
import time
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log
from .client import SplitNNClient


class ClientManager(MessageManager):
    """
    args里面要有MPI的 comm, rank, max_rank(也就是comm.size()-1) 其他的暂时不用
    trainer就是SplitNNClient的一个实例
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client",
                         args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)
        self.last = -1

    def run(self):
        # logging.info("client model {}".format(self.trainer.model))
        # logging.info("{} begin run_forward_pass".format(self.trainer.rank))

        self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        # logging.info("acts:{}  labels:{}".format(acts.shape, labels))
        # logging.info("{} end run_forward_pass act :{}".format(self.trainer.rank, acts.shape))
        logging.info("rank {}".format(self.trainer.rank))
        self.send_activations_and_labels_to_server(
            acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            logging.warning("validate {}".format(i))
            self.run_forward_pass()
            while True:
                if self.com_manager.q_receiver.qsize() > 0:
                    msg_params = self.com_manager.q_receiver.get()
                    self.com_manager.notify(msg_params)
                    break
                else:
                    time.sleep(0.1)
        self.trainer.write_log()

        if self.args["save_model_epoch"] > 0 and self.trainer.epoch_count % self.args["save_model_epoch"] == 0:
            self.save_model(self.trainer.epoch_count)

        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.trainer.train_mode()
            self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_START_VALIDATION,
                                              self.handle_validation_start_sign)

    def handle_message_gradients(self, msg_params):
        tot, cor, vl = msg_params.get(MyMessage.MSG_AGR_KEY_RESULT)
        logging.info("tot,cor,vl: {}, {}, {}".format(tot, cor, vl))
        logging.info("self.trainer.phase: {}".format(self.trainer.phase))
        self.trainer.total += tot
        self.trainer.correct += cor
        self.trainer.val_loss += vl
        self.trainer.step += 1
        if self.trainer.phase == "train":
            self.trainer.write_log()
            grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
            self.trainer.backward_pass(grads)
            logging.warning("batch: {} len {}".format(
                self.trainer.batch_idx, len(self.trainer.trainloader)))
            # if self.trainer.rank == 2 and self.trainer.batch_idx == len(self.trainer.trainloader) // 2:
            #     self.run_eval()

            if self.trainer.batch_idx == len(self.trainer.trainloader):
                # torch.save(self.trainer.model, self.args["model_tmp_path"])
                self.trainer.print_com_size(self.com_manager)
                self.send_validation_sign(0)
                # self.run_eval()
            else:
                self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        logging.warning("acts to {} phase {}".format(
            receive_id, self.trainer.phase))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                          self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_EPOCH,
                           self.trainer.epoch_count)

        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def send_validation_sign(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION,
                          self.rank, receive_id)
        self.send_message(message)

    def handle_validation_start_sign(self, msg_params):

        self.run_eval()

    def save_model(self, epoc):
        torch.save(self.trainer.model,
                   "./model_save/attack_acts/PSL/{}/C_{}_A_{}_E_{}.pkl".format(self.args["dataset"], self.trainer.rank, self.args["partition_alpha"], epoc))
