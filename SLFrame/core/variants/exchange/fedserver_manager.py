import random
import numpy as np
import logging
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log


class FedServerManager(MessageManager):

    def __init__(self, args, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.log = Log(self.__class__.__name__, args)
        self.client_num = args["max_rank"] - 1
        self.clients_model = {}
        self.counter = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                                              self.handle_message_model_from_client)

    def handle_message_model_from_client(self,msg_params):
        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.clients_model[client_id]=model
        self.counter += 1
        logging.warning("fed server recv model")
        if self.counter==self.client_num:
            lst= self.get_sending_list("random_permutation")
            self.counter = 0

            for i in range(self.client_num):
                self.send_message_model_to_client(i+1,self.clients_model[lst[i]])

    def get_sending_list(self,mode):
        # return a list in which each element is in [1,client_num]
        tmp = []
        if mode == "random_permutation":
            tmp = [i+1 for i in range(self.client_num)]
            random.shuffle(tmp)

        elif mode == "cycle":
            tmp = [(i+1)%self.client_num + 1 for i in range(self.client_num)]
        elif mode == "random_indepentent":
            tmp = [random.randint(1,self.client_num) for i in range(self.client_num)]
        logging.warning(tmp)
        return tmp

    def send_message_model_to_client(self,client_id,models):
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODELS, self.rank, client_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL,models)
        self.send_message(message)



