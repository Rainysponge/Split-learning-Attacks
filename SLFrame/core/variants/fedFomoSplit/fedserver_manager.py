import random
import numpy as np
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log


class FedServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.client_num = args["max_rank"] - 1
        self.probability_mat = np.mat(np.ones(self.client_num,self.client_num))
        self.clients_model = {}
        self.counter = 0
        self.max_model_num= args["max_model_num"]
        self.mapping_table = {}
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                                              self.handle_message_model_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_WEIGHTS,
                                              self.handle_message_weights_from_client)

    def handle_message_model_from_client(self,msg_params):
        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.clients_model[client_id]=model
        self.counter += 1
        if self.counter==self.client_num:
            for i in range(self.client_num):
                models = self.randomly_select_model_by_probmat(i)
                self.send_message_models_to_client(i,models)


    def handle_message_weights_from_client(self,msg_params):
        client_id= msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)



    def send_message_models_to_client(self,client_id,models):
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODELS, self.rank, client_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODELS,models)
        self.send_message(message)

    """
        对客户端隐藏发送的模型的实际编号, 有利于保护隐私
        发给client_id的tmp里的编号x(0-indexed)实际上是mapping_table[client_id]的第x元素的值(1-indexed)
    """




