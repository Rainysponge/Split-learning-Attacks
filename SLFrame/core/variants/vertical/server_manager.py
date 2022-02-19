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
        self.active_node = -1
        self.finished_nodes = 0
        self.log = Log(self.__class__.__name__, args)
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        message.add_params(MyMessage.MSG_AGR_KEY_RESULT,
                           (self.trainer.total, self.trainer.correct, self.trainer.val_loss))
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        if client_phase == "train":
            self.trainer.train_mode()
            self.trainer.add_client_local_result(sender, acts)

        else:
            self.trainer.eval_mode()
            self.trainer.add_client_local_result(sender, acts)
        all_received = self.trainer.check_whether_all_receive()
        if all_received:
            self.trainer.forward_pass(acts, labels)

            grads = None
            if self.trainer.phase == "train":
                grads = self.trainer.backward_pass()

            grads_list = grads.split([5, 5], dim=1)
            for idx in range(self.trainer.client_number):
                self.send_grads_to_client(idx+1, grads_list[idx])
        else:
            pass


    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()
