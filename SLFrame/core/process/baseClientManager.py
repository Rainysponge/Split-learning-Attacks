import logging
import torch
import time
import abc
# from message_define import MyMessage
from ..communication.msg_manager import MessageManager
from ..communication.message import Message
from ..log.Log import Log
from .baseClient import baseClient


class baseClientManager(MessageManager):
    """
    args里面要有MPI的 comm, rank, max_rank(也就是comm.size()-1) 其他的暂时不用
    trainer就是SplitNNClient的一个实例
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        # self.trainer = type(SplitNNClient)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)


    @abc.abstractmethod
    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        logging.info("{} end run_forward_pass act :{}".format(self.trainer.rank, acts.shape))
        logging.warning("rank {}".format(self.trainer.rank))
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    @abc.abstractmethod
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

        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.trainer.train_mode()
            self.run_forward_pass()


    @abc.abstractmethod
    def handle_message_gradients(self, msg_params):
        pass

    @abc.abstractmethod
    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        pass

    @abc.abstractmethod
    def send_finish_to_server(self, receive_id):
        pass

    @abc.abstractmethod
    def send_validation_sign(self, receive_id):
        pass

    @abc.abstractmethod
    def handle_validation_start_sign(self, msg_params):
        self.run_eval()
