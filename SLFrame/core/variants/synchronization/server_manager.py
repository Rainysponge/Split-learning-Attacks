from mpi4py import MPI
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log


class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.active_node = -1
        self.finished_nodes = 0
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
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION,
                                              self.handle_validation_sign)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)

        self.trainer.gradients_dict[sender] = (acts, labels, client_phase)
        self.trainer.gradients_number += 1

        if self.trainer.gradients_number == self.trainer.client_number:
            for i in range(1, self.trainer.client_number + 1):
                acts, labels, client_phase = self.trainer.gradients_dict[i]

                if client_phase == "train":
                    self.trainer.train_mode()
                else:
                    self.trainer.eval_mode()
                self.trainer.forward_pass(acts, labels)
                # self.log.info(acts.shape)
                # self.log.info(type(acts))

                grads = None
                if self.trainer.phase == "train":
                    grads = self.trainer.backward_pass()

                self.send_grads_to_client(i, grads)
            self.trainer.gradients_number = 0
            for i in range(1, self.trainer.client_number + 1):
                self.send_next_batch_sign(i)

    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def handle_validation_sign(self, msg_params):
        self.trainer.validation_sign_number += 1
        if self.trainer.validation_sign_number == self.trainer.MAX_RANK:
            self.trainer.validation_sign_number = 0
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_validation_sign_to_client(idx)

    def send_validation_sign_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_START_VALIDATION, 0, receive_id)
        self.send_message(message)

    def send_next_batch_sign(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_NEXT, 0, receive_id)
        self.send_message(message)
