import logging
import torch
import time
import sys
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

        self.log = Log(self.__class__.__name__, args)

    def run(self):
        # if self.rank == 1:
        self.trainer.train_mode()
        self.run_forward_pass()
        super().run()  # run 要放在最后是因为先要发送信息之后才能开始监听.. 不然大家都在等别人的消息

    def run_forward_pass(self):
        acts = self.trainer.forward_pass()
        logging.warning("{} send acts to server".format(self.trainer.rank))
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        acts = self.trainer.forward_pass()
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)

        # for i in range(len(self.trainer.dataloader)):
        #     logging.warning("validate {} from {} len {}".format(i, self.trainer.rank, len(self.trainer.dataloader)))
        #     self.run_forward_pass()
        #     while True:
        #         if self.com_manager.q_receiver.qsize() > 0:
        #             msg_params = self.com_manager.q_receiver.get()
        #             self.com_manager.notify(msg_params)
        #             break
        #         else:
        #             time.sleep(0.5)
        # self.trainer.write_log()
        # self.trainer.epoch_count += 1
        # if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
        #     self.send_finish_to_server(self.trainer.SERVER_RANK)
        #     self.finish()
        # else:
        #     # self.trainer.train_mode()
        #     # self.run_forward_pass()
        #     self.send_test_end_to_server(self.trainer.SERVER_RANK)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_TEST,
                                              self.handle_test_sign)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_ACTS,
                                              self.handle_message_acts_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_TRAIN,
                                              self.handle_train_sign)

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        if grads is not None:
            self.trainer.backward_pass(type=0, grads=grads)
        logging.warning("batch: {} len {} from {}".format(self.trainer.batch_idx, len(self.trainer.dataloader),
                                                          self.trainer.rank))
        if self.trainer.batch_idx == len(self.trainer.dataloader):
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.trainer.epoch_count))
            if self.trainer.phase == 'validation':

                self.trainer.write_log()
                self.trainer.epoch_count += 1
                if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE \
                        and self.trainer.rank == self.trainer.MAX_RANK:
                    self.send_finish_to_server(self.trainer.SERVER_RANK)
                    self.finish()
                self.send_test_end_to_server(self.trainer.SERVER_RANK)
            else:
                self.send_train_end_to_server(self.trainer.SERVER_RANK)

            # self.run_eval()
        else:

            self.run_forward_pass()

    def handle_message_acts_from_server(self, msg_params):
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        # logging.warning("QAQ3")
        self.trainer.forward_pass(type=1, inputs=acts)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass(type=1)
            self.send_grads_to_server(self.trainer.SERVER_RANK, grads)
        else:
            self.send_grads_to_server(self.trainer.SERVER_RANK, None)

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_to_server(self, acts, receive_id):
        #    logging.warning("{} acts to {}".format(self.rank,receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)

        message.add_params(MyMessage.MSG_ARG_KEY_BATCH_END, self.trainer.batch_idx + 1 == len(self.trainer.dataloader))
        self.send_message(message)

    def send_grads_to_server(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        logging.warning("client{} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def handle_test_sign(self, msg_params):
        logging.info("{} handle_test_sign ".format(self.trainer.rank))
        self.trainer.eval_mode()
        self.run_forward_pass()

    def send_test_end_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_TEST_EMD, self.rank, receive_id)
        self.send_message(message)

    def send_train_end_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_TRAIN_END, self.rank, receive_id)
        self.send_message(message)

    def handle_train_sign(self, msg_params):
        logging.info("{} handle_train_sign ".format(self.trainer.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

