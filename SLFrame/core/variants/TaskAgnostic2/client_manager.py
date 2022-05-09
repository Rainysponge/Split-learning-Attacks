import logging
import torch
import time
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log
from .client import SplitNNClient


class ClientManager(MessageManager):
    """
    args里面要有MPI的 comm, rank, max_rank(也就是comm.size()-1) 其他的暂时不用
    trainer就是SplitNNClient的一个实例
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        # self.trainer = type(SplitNNClient)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)

    def run(self):
        # logging.info("{} begin run_forward_pass".format(self.trainer.rank))
        if self.trainer.args["dataset_cur"] == 0:
            self.trainer.train_mode()
            self.run_forward_pass()
        super().run()

    def run_forward_pass(self):
        acts = self.trainer.forward_pass()
        logging.warning("rank {} send acts to server".format(self.args["rank"]))
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            logging.warning("validate {}".format(i))
            self.run_forward_pass()
            while True:
                if self.com_manager.q_receiver.qsize() > 0:
                    msg_params = self.com_manager.q_receiver.get()
                    self.com_manager.notify(msg_params)
                    break
                else:
                    time.sleep(0.5)
        self.trainer.write_log()
        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            if self.args['client_split'][self.args['dataset_cur']] == self.args['rank']:

                next_group_cur = (self.args['dataset_cur'] + 1) % len(self.args['client_split'])
                logging.warning("next_group_cur {}".format(next_group_cur))
                start = 1 if next_group_cur == 0 else self.args['client_split'][next_group_cur - 1] + 1
                end = self.args['client_split'][next_group_cur] + 1

                for i in range(start, end):
                    self.send_semaphore_to_client(i)


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_ACTS,
                                              self.handle_message_acts_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEMAPHORE,
                                              self.handle_message_semaphore_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MODEL,
                                              self.handle_message_model_param_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_READY_TO_GET_MODEL,
                                              self.handle_ready_to_get_model)

    def handle_message_model_param_from_server(self, msg_params):
        model_param = msg_params.get(MyMessage.MSG_AGR_HEAD_MODEL)
        # self.log.info("rank {} get model param".format(self.args['rank']))
        self.trainer.model.load_state_dict(model_param)
        model_param = msg_params.get(MyMessage.MSG_AGR_TAIL_MODEL)
        # self.log.info(model_param["block1.0.weight"])
        self.trainer.model_2.load_state_dict(model_param)
        self.run_eval()

    # MSG_TYPE_S2C_SEMAPHORE
    def handle_message_semaphore_from_server(self, msg_params):
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        self.trainer.step += 1
        if self.trainer.phase == "train":
            self.trainer.write_log()
            grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
            self.trainer.backward_pass(type=0, grads=grads)
            logging.warning("rank: {} batch: {} len {}".format(self.args['rank'], self.trainer.batch_idx,
                                                               len(self.trainer.trainloader)))

            if self.trainer.batch_idx == len(self.trainer.trainloader):
                # torch.save(self.trainer.model, self.args["model_tmp_path"])
                self.send_ready_to_send_model(0)
            else:
                if self.args["rank"] == self.args['client_split'][self.args['dataset_cur']]:
                    self.send_next_batch_to_server(0)
        else:
            self.run_forward_pass()

    # def send_activations_and_labels_to_server(self, acts, labels, receive_id):
    #     logging.warning("acts to {}".format(receive_id))
    #     message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
    #     message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
    #     self.send_message(message)

    def send_activations_to_server(self, acts, receive_id):
        #    logging.warning("{} acts to {}".format(self.rank,receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])

        self.send_message(message)

    def handle_message_acts_from_server(self, msg_params):
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        # logging.warning("QAQ3")
        self.trainer.forward_pass(type=1, inputs=acts)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass(type=1)
            self.send_grads_to_server(self.trainer.SERVER_RANK, grads)

    # def handle_message_acts_from_server(self, msg_params):
    def handle_message_semaphore(self, msg_params):
        logging.warning("client {} recv semapgore".format(self.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

    def send_model_param_to_fed_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_AGR_HEAD_MODEL, self.trainer.model.state_dict())
        message.add_params(MyMessage.MSG_AGR_TAIL_MODEL, self.trainer.model_2.state_dict())
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def send_grads_to_server(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)

        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.rank, receive_id)
        self.send_message(message)

    def send_next_batch_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_NEXT_BATCH, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        self.send_message(message)

    # MSG_TYPE_C2S_READY_TO_SEND_MODEL
    def send_ready_to_send_model(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_READY_TO_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        self.send_message(message)

    # MSG_TYPE_S2C_READY_TO_GET_MODEL
    def handle_ready_to_get_model(self, msg_params):
        self.send_model_param_to_fed_server(0)
