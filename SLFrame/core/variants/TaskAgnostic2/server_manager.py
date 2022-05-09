from mpi4py import MPI
import logging
import time
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
        self.sender_list = dict()
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        # message.add_params(MyMessage.MSG_AGR_KEY_RESULT,
        #                    (self.trainer.total, self.trainer.correct, self.trainer.val_loss))
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRADS,
                                              self.handle_message_grads)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_MODEL,
                                              self.handle_message_model_param)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_NEXT_BATCH,
                                              self.handle_next_batch)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_READY_TO_SEND_MODEL,
                                              self.handle_model_ready_sign)

    # def handle_message_acts(self, msg_params):
    #     acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
    #     self.active_node = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
    #     client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
    #     if client_phase == "train":
    #         self.trainer.train_mode()
    #     else:
    #         self.trainer.eval_mode()
    #     self.trainer.forward_pass(acts, labels)
    #     # self.log.info(acts.shape)
    #     # self.log.info(type(acts))
    #
    #
    #
    #     grads = None
    #     if self.trainer.phase == "train":
    #         grads = self.trainer.backward_pass()
    #
    #     self.send_grads_to_client(self.active_node, grads)
    def handle_message_acts(self, msg_params):
        # logging.warning("server recv acts")
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        self.trainer.client_act_dict[sender] = acts
        # self.trainer.client_act_check_dict[dataset_cur]
        if dataset_cur in self.trainer.client_act_check_dict:
            self.trainer.client_act_check_dict[dataset_cur] += 1
        else:
            self.trainer.client_act_check_dict[dataset_cur] = 1
        # 这里要判断是否达到该数据集下的所有客户端
        if self.trainer.client_act_check_dict[dataset_cur] == client_number:
            start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
            end = self.args['client_split'][dataset_cur] + 1
            # 先前传播

            for i in range(start, end):
                acts = self.trainer.client_act_dict[i]
                acts2 = self.trainer.forward_pass(acts)
                self.send_acts_to_client(i, acts2)
                # tmp = self.com_manager.q_receiver.qsize()
                if phase == "train":
                    while True:
                        # 等待梯度回传
                        if self.com_manager.q_receiver.qsize() > 0:
                            msg_params = self.com_manager.q_receiver.get()
                            if msg_params.get(MyMessage.MSG_ARG_KEY_TYPE) == MyMessage.MSG_TYPE_C2S_SEND_GRADS:
                                logging.info(
                                    "server get grads from {} key {}".format(
                                        msg_params.get(MyMessage.MSG_ARG_KEY_SENDER),
                                        msg_params.get(MyMessage.MSG_ARG_KEY_TYPE))
                                )
                                self.com_manager.notify(msg_params)
                                break
                        else:
                            time.sleep(0.5)

            self.trainer.client_act_check_dict[dataset_cur] = 0

    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def handle_message_model_param(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        head_model_param = msg_params.get(MyMessage.MSG_AGR_HEAD_MODEL)
        tail_model_param = msg_params.get(MyMessage.MSG_AGR_TAIL_MODEL)
        sample_number = msg_params.get(MyMessage.MSG_AGR_KEY_SAMPLE_NUM)
        client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        self.trainer.sum_sample_number += sample_number
        self.trainer.model_param_dict[sender] = (sample_number, head_model_param, tail_model_param)

        self.sender_list[sender] = True
        # self.trainer.client_sample_dict[sender] = sample_number
        self.trainer.model_param_num += 1
        if self.trainer.model_param_num == client_number:
            self.log.info(self.sender_list)
            # self.log.info("get all model params ---- from rank {}".format(self.trainer.rank))
            self.trainer.model_param_num = 0
            head_model_avg = head_model_param
            tail_model_avg = tail_model_param
            start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
            end = self.args['client_split'][dataset_cur] + 1
            for key in head_model_param.keys():
                for idx in range(start, end):

                    self.sender_list[idx] = False

                    local_sample_number, local_model_params, _ = self.trainer.model_param_dict[idx]
                    w = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        head_model_avg[key] = local_model_params[key] * w
                    else:
                        head_model_avg[key] += local_model_params[key] * w

            for key in tail_model_param.keys():
                for idx in range(start, end):

                    self.sender_list[idx] = False

                    local_sample_number, _, local_model_params = self.trainer.model_param_dict[idx]
                    w = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        tail_model_avg[key] = local_model_params[key] * w
                    else:
                        tail_model_avg[key] += local_model_params[key] * w
            self.trainer.sum_sample_number = 0
            # self.log.info(head_model_avg)
            # self.log.info(tail_model_avg)
            for idx in range(start, end):
                # self.log.info("send_model_param_to_fed_client： {}".format(idx))
                self.send_model_param_to_fed_client(idx, head_model_avg, tail_model_avg)

    # avg server
    def send_model_param_to_fed_client(self, receive_id, model_head_param, model_tail_param):
        message = Message(MyMessage.MSG_TYPE_S2C_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_AGR_TAIL_MODEL, model_tail_param)
        message.add_params(MyMessage.MSG_AGR_HEAD_MODEL, model_head_param)
        self.send_message(message)

    def send_acts_to_client(self, receive_id, acts):
        # logging.warning("server acts2 to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        self.send_message(message)

    def handle_message_grads(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        if phase == "train":
            new_grad = self.trainer.backward_pass(grads)
            self.send_grads_to_client(sender, new_grad)
        else:
            self.send_grads_to_client(sender, None)

    # MSG_TYPE_S2C_SEMAPHORE
    def send_semaphore_to_clients(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_SEMAPHORE, self.rank, receive_id)
        message.add_params("SEMAPHORE", 0)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        # logging.warning("client{} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))

        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    # MSG_TYPE_S2C_READY_TO_GET_MODEL
    def send_ready_get_model(self, receive_id):
        # logging.warning("client{} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))

        message = Message(MyMessage.MSG_TYPE_S2C_READY_TO_GET_MODEL, self.rank, receive_id)
        self.send_message(message)

    def handle_next_batch(self, msg_params):
        # client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
        end = self.args['client_split'][dataset_cur] + 1
        for i in range(start, end):
            # self.log.info("send sign to rank {}".format(i))
            self.send_semaphore_to_clients(i)

    def handle_model_ready_sign(self, msg_params):
        # 记录个数
        client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        self.trainer.model_ready_num += 1

        if self.trainer.model_ready_num == client_number:
            self.trainer.model_ready_num = 0
            start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
            end = self.args['client_split'][dataset_cur] + 1
            for i in range(start, end):
                # self.log.info("all client have been trained one epoch".format(i))
                self.send_ready_get_model(i)
