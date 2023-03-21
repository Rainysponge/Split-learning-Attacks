from mpi4py import MPI
import logging
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log
import torch


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
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                                              self.handle_message_model_param)

    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def get_dict_n2_dist(x1: dict, x2: dict):
        pass

    def handle_message_model_param(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        sample_number = msg_params.get(MyMessage.MSG_AGR_KEY_SAMPLE_NUM)
        if self.trainer.last_param is None:
            self.trainer.last_param = model_param
        self.trainer.sum_sample_number += sample_number
        self.trainer.model_param_dict[sender] = (sample_number, model_param)
        # self.sender_list[sender] = True
        # self.trainer.client_sample_dict[sender] = sample_number
        self.trainer.model_param_num += 1
        if self.trainer.model_param_num == self.trainer.MAX_RANK:
            # self.log.info(self.sender_list)
            # for key in self.trainer.model_param_dict.keys():
            #     logging.info("key: ".format(key))
            self.log.info(
                "get all model params ---- from rank {}".format(self.trainer.rank))

            self.trainer.model_param_num = 0
            model_avg = model_param
            dist = 0
            t_n = 0
            for key in model_avg.keys():
                for idx in range(1, self.trainer.MAX_RANK + 1):

                    # self.sender_list[idx] = False

                    local_sample_number, local_model_params = self.trainer.model_param_dict[idx]
                    w = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        model_avg[key] = local_model_params[key] * w
                    else:
                        model_avg[key] += local_model_params[key] * w
                #self.log.info("key:{}\n value:{}".format(key, model_avg[key]))
                dif = self.trainer.last_param[key]-model_avg[key]
                sz = 1
                for v in dif.shape:
                    sz *= v
                t_n += sz
                dist += torch.pow(dif, 2).sum(list(range(len(dif.shape))))

            self.log.info("Dist:{d}".format(d=dist/t_n))
            self.trainer.sum_sample_number = 0
            self.trainer.last_param = model_avg
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_model_param_to_fed_client(idx, model_avg)

    def send_model_param_to_fed_client(self, receive_id, model_avg_param):
        message = Message(MyMessage.MSG_TYPE_S2C_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, model_avg_param)
        self.send_message(message)
