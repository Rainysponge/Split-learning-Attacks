import logging
import torch
import time
import abc
# from message_define import MyMessage
from ..communication.msg_manager import MessageManager


class baseServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        # self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.active_node = -1
        self.finished_nodes = 0
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    @abc.abstractmethod
    def send_grads_to_client(self, receive_id, grads=None):
        pass

    @abc.abstractmethod
    def handle_message_acts(self, msg_params):
        pass

    @abc.abstractmethod
    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    @abc.abstractmethod
    def handle_validation_sign(self, msg_params):
        pass

    @abc.abstractmethod
    def send_validation_sign_to_client(self, receive_id):
        pass
