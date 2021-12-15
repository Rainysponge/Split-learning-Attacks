from ..communication.msg_manager import MessageManager
from ..communication.message import Message
from ..communication.message_define import MyMessage


class ClientManager(MessageManager):

    def __init__(self,arg, trainer, backend="MPI"):
        super().__init__(arg, "client", arg["comm"], arg["rank"], arg["max_rank"] + 1, backend)
        self.trainer = trainer
        self.trainer.train_mode()
        self.round_idx = 0

    def run(self):
        if self.trainer.rank == 1:
            # logging.info("Starting protocol from rank 1 process")
            self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.round_idx += 1
        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
                self.send_finish_to_server(self.trainer.SERVER_RANK)
        else:
            self.send_semaphore_to_client(self.trainer.node_right)

        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE:
            self.finish()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)

    def handle_message_semaphore(self, msg_params):
        # no point in checking the semaphore message
        self.trainer.train_mode()
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            self.round_idx += 1
            self.run_eval()
        else:
            self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

