import logging
import torch
import time
from queue import Queue
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log


class ClientManager(MessageManager):
    """
    args里面要有MPI的 comm, rank, max_rank(也就是comm.size()-1) 其他的暂时不用
    trainer就是SplitNNClient的一个实例
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)
        self.server_ready = 0
        self.acts_queue = Queue(0)
        self.labels_queue = Queue(0)
        self.round_idx = 0

    def run(self):
        if self.rank == 1:
            self.server_ready = 1
            # self.run_forward_pass()
        # self.log.info("self.rank {}".format(self.rank))
        self.trainer.train_mode()
        # logging.info("rank {} training. The length is {}".format(self.trainer.rank, len(self.trainer.trainloader)))
        self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        if self.server_ready == 0:
            logging.info("rank {} is training".format(self.trainer.rank))
            acts, labels = self.trainer.forward_pass()
            while labels is not None:
                # logging.info("rank {} training. The length is {}".format(self.trainer.rank, len(self.acts_list)))
                # logging.info("rank {} training. The acts is {}".format(self.trainer.rank, acts))

                self.acts_queue.put(acts)
                self.labels_queue.put(labels)
                acts, labels = self.trainer.forward_pass()

        elif self.server_ready == 1:
            if self.acts_queue.empty():
                acts, labels = self.trainer.forward_pass()
                if labels is not None:
                    self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
                    self.trainer.batch_idx += 1
            else:
                acts, labels = self.acts_queue.get(), self.labels_queue.get()
                self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
                self.trainer.batch_idx += 1
                # for act, label in zip(self.acts_list, self.labels_list):

        # while self.server_ready == 0:
        #     logging.info("rank {} is waiting".format(self.trainer.rank))

        # acts, labels = self.trainer.forward_pass()

        # if self.server_ready == 0:
        #     logging.info("rank {} is trainning. The length is {}".format(self.trainer.rank, len(self.acts_list)))
        #
        # elif self.server_ready == 1 and len(self.acts_list) > 0:
        #     for act, label in zip(self.acts_list, self.labels_list):
        #         self.send_activations_and_labels_to_server(act, label, self.trainer.SERVER_RANK)
        #         self.trainer.batch_idx += 1

        # elif self.server_ready == 1:
        #     self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()

        for i in range(len(self.trainer.testloader)):
            logging.warning("rank {} validate {}".format(self.trainer.rank, i))
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.round_idx += 1
        self.log.info(
            "noderight {} self {} max_rank {}".format(self.trainer.node_right, self.trainer.rank,
                                                      self.trainer.MAX_RANK))
        self.log.info("round {} max_epoch {}".format(self.round_idx, self.trainer.MAX_EPOCH_PER_NODE))
        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            logging.warning("rank {} finish".format(self.trainer.rank))
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.server_ready = 0
            logging.info("send_semaphore_to_client {}".format(self.trainer.node_right))
            self.send_semaphore_to_client(self.trainer.node_right)

        self.trainer.batch_idx = 0
        self.trainer.train_mode()
        self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        # self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_READY,
        #                                       self.handle_message_to_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)

    def handle_message_semaphore(self, msg_params):
        # no point in checking the semaphore message
        logging.warning("client {} recv sema".format(self.rank))
        # for act, label in
        self.server_ready = 1
        # self.trainer.train_mode()
        self.run_forward_pass()

    # def handle_message_to_server(self, msg_params):
    #     # self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
    #     self.server_ready = 1

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        logging.warning("rank: {} batch: {} len {}".format(self.trainer.rank, self.trainer.batch_idx,
                                                           len(self.trainer.trainloader)))
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.round_idx))
            self.run_eval()
        else:
            self.run_forward_pass()

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        logging.warning("acts to {}".format(receive_id))
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
        logging.warning("client {} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)
