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

        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                                              self.handle_message_validation_over)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRADS,
                                              self.handle_message_grads)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_TEST_EMD,
                                              self.handle_test_end_from_clients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_TRAIN_END,
                                              self.handle_train_end_from_clients)

    def handle_message_acts(self, msg_params):
        # logging.warning("server recv acts")
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        batch_end = msg_params.get(MyMessage.MSG_ARG_KEY_BATCH_END)
        phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        if phase == "train":
            self.trainer.train_mode()
        else:
            self.trainer.eval_mode()
            # logging.warning("server recv acts {}".format(self.trainer.is_waiting))
        # 无论如何都先把(sender, acts) 存入到queue中
        self.trainer.client_act_queue.put((sender, acts, batch_end))
        if not self.trainer.is_waiting:
            # 先前传播
            # logging.warning("forwordpass")
            self.trainer.is_waiting = True
            sender, acts, batch_end = self.trainer.client_act_queue.get()
            acts2 = self.trainer.forward_pass(acts)
            self.send_acts_to_client(sender, acts2, batch_end)
        # else:
        #     acts2 = self.trainer.forward_pass(acts)
        #     self.send_acts_to_client(sender, acts2, batch_end)

    def handle_message_grads(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        if grads is not None:
            new_grad = self.trainer.backward_pass(grads)

            self.send_grads_to_client(sender, new_grad)
            if self.trainer.client_act_queue.empty():
                # 为空
                self.trainer.is_waiting = False
            else:
                # 不为空
                sender, acts, batch_end = self.trainer.client_act_queue.get()

                acts2 = self.trainer.forward_pass(acts)
                self.send_acts_to_client(sender, acts2, batch_end)

                # if self.trainer.client_batch_end_num == self.trainer.client_number:
                #
                #     for i in range(1, self.trainer.client_number + 1):
                #         self.send_test_sign_to_client(i)
                #     self.trainer.client_batch_end_num = 0
                #     self.trainer.is_waiting = False
        else:
            self.send_grads_to_client(sender, None)
            if self.trainer.client_act_queue.empty():
                # 为空
                self.trainer.is_waiting = False
            else:
                # 不为空
                sender, acts, batch_end = self.trainer.client_act_queue.get()

                if batch_end:
                    self.trainer.client_batch_end_num += 1
                acts2 = self.trainer.forward_pass(acts)

                self.send_acts_to_client(sender, acts2, batch_end)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        # logging.warning("over")
        self.trainer.validation_over()

    def handle_message_finish_protocol(self):
        self.finish()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def send_acts_to_client(self, receive_id, acts, batch_end):
        # logging.warning("server acts2 to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        # message.add_params(MyMessage.MSG_ARG_KEY_BATCH_END, batch_end)
        self.send_message(message)

    def send_test_sign_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_TEST, self.rank, receive_id)
        self.send_message(message)

    def handle_test_end_from_clients(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # logging.info("{} end".format(sender))
        self.trainer.client_test_end_num += 1
        if self.trainer.client_test_end_num == self.trainer.client_number:
            for i in range(1, self.trainer.client_number + 1):
                self.send_train_sign_to_clients(i)
            self.trainer.client_test_end_num = 0

    def handle_train_end_from_clients(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # logging.info("{} train end".format(sender))
        self.trainer.client_train_end_num += 1
        # logging.info("{}, {}".format(self.trainer.client_train_end_num, self.trainer.client_number))
        if self.trainer.client_train_end_num == self.trainer.client_number:
            # logging.info("queue len {}".format(self.trainer.client_act_queue.qsize()))
            self.trainer.is_waiting = False
            for i in range(1, self.trainer.client_number + 1):
                self.send_test_sign_to_client(i)
            self.trainer.client_train_end_num = 0

    def send_train_sign_to_clients(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_TRAIN, self.rank, receive_id)
        self.send_message(message)
