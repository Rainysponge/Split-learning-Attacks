from mpi4py import MPI
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
import logging
from ...log.Log import Log

class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.round_idx = 0
        self.log = Log(self.__class__.__name__, args)

        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                                              self.handle_message_validation_over)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                                              self.handle_message_client_model)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_NEXT_CLIENT,
                                              self.handle_next_client)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.trainer.forward_pass(acts, labels)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            if self.trainer.state == "A":
                self.send_grads_to_client(self.trainer.active_node, grads)
            else:
                self.send_grads_to_client(self.trainer.active_node, None)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        # logging.warning("over")
        self.trainer.validation_over()

    def handle_message_finish_protocol(self, msg_params):
        self.finish()

    def handle_message_client_model(self, msg_params):
        model = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.trainer.client_model = model

    def send_model_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.client_model)
        message.add_params(MyMessage.MSG_ARG_KEY_STATE, self.trainer.state)
        self.send_message(message)

    def handle_next_client(self, msg_params):
        next_client = msg_params.get(MyMessage.MSG_ARG_KEY_NEXT_CLIENT)
        if next_client == 1:
            # new turn
            if self.trainer.state == "A":
                self.trainer.last_update_loss = self.trainer.total_loss / 156 / 3
            self.trainer.delta_loss = self.trainer.last_update_loss - self.trainer.total_loss / 156 / 3
            self.log.info("loss:{}, {}".format(self.trainer.last_update_loss, self.trainer.total_loss / 156 / 3))
            if self.trainer.delta_loss >= self.trainer.loss_thred:
                self.trainer.state = "A"
            else:
                if self.trainer.state == "A":
                    self.trainer.state = "B"
                else:
                    self.trainer.state = "C"
            self.trainer.total_loss = 0
        self.send_model_to_client(next_client)

    # def send_message_to_next_client(self, receive_id):
    #     message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODEL, self.rank, receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.client_model)
    #     self.send_message(message)
