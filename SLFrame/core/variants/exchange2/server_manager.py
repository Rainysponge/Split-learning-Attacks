from mpi4py import MPI
import random
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
        self.client_num = args["max_rank"]
        self.grads_buffer= {}
        self.result_buffer= {}
        self.counter=0
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        message.add_params(MyMessage.MSG_AGR_KEY_RESULT,
                          self.result_buffer[receive_id])
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.active_node = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
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

        self.result_buffer[self.active_node]= (self.trainer.total, self.trainer.correct, self.trainer.val_loss)
        self.grads_buffer[self.active_node]=grads
        if grads is not None:
            logging.warning("grads from {}: {}".format(self.active_node,sum(sum(sum(sum(grads))))))
        self.counter += 1
        if self.counter == self.client_num:
            self.counter = 0
            sending_list = self.get_sending_list("random_permutation")
            for i in range(self.client_num):
                self.send_grads_to_client(i+1,self.grads_buffer[sending_list[i]])
                if self.grads_buffer[sending_list[i]] is not None:
                    logging.warning("send to client {} : {}".format(i+1,sum(sum(sum(sum(self.grads_buffer[sending_list[i]]))))))



    def get_sending_list(self, mode):
        # return a list in which each element is in [1,client_num]
        tmp = []
        if mode == "random_permutation":
            tmp = [i+1 for i in range(self.client_num)]
            random.shuffle(tmp)

        elif mode == "cycle":
            tmp = [(i+1)%self.client_num + 1 for i in range(self.client_num)]
        elif mode == "random_indepentent":
            tmp = [random.randint(1,self.client_num) for i in range(self.client_num)]

        import logging
        logging.warning(tmp)
        return tmp




    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()
