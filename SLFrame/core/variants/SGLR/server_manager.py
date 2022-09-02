from mpi4py import MPI
import logging
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
                           (self.trainer.res_dict[receive_id]))
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
        grads = None
        self.trainer.act_dict[sender] = (acts, labels)
        self.trainer.act_number += 1
        # self.log.info("act_number: {} MAX_RANK: {}".format(self.trainer.act_number, self.trainer.MAX_RANK))
        if self.trainer.act_number == self.trainer.MAX_RANK:
            if client_phase == "train":
                self.trainer.train_mode()
                self.trainer.act_number = 0

                for i in range(1, self.trainer.MAX_RANK + 1):
                    self.trainer.forward_pass(self.trainer.act_dict[i][0], self.trainer.act_dict[i][1], i)

                    grads = self.trainer.backward_pass()

                    self.trainer.add_client_local_grads(i, grads)
                    all_receive, active_list = self.trainer.check_whether_all_receive()
                    if all_receive:
                        # 发回
                        logging.info("active_list: {}".format(active_list))
                        grads = self.trainer.splitAvg(active_list)
                        # self.log.info(grads.shape)
                        for idx in range(1, self.trainer.client_number+1):
                            if idx in active_list:
                                self.send_grads_to_client(idx, grads)
                            else:
                                self.send_grads_to_client(idx, self.trainer.client_train_grads_list[idx])
                self.trainer.optimizer.step()
            else:
                self.trainer.act_number = 0
                self.trainer.eval_mode()
                for i in range(1, self.trainer.MAX_RANK + 1):
                    self.trainer.forward_pass(acts, labels, i)
                    self.send_grads_to_client(i, grads)

    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()
