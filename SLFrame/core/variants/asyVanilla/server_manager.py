from mpi4py import MPI
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
import logging


class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.round_idx = 0
        self.finish_number = 0
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def register_message_receive_handlers(self):
        # dict
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                                              self.handle_message_validation_over)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # logging.info("sender is {}".format(sender))
        self.trainer.forward_pass(acts, labels)
        self.trainer.active_node = sender
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            logging.info("send_grads_to_clients {}".format(sender))
            self.send_grads_to_client(self.trainer.active_node, grads)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        self.trainer.validation_over()

    def handle_message_finish_protocol(self, msg_params):
        self.finish_number += 1
        logging.info("finish {}".format(self.finish_number))
        if self.finish_number == self.trainer.args["client_number"]:
            self.finish()
