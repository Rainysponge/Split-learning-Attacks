from .observer import Observer
from .mpi.mpi_com_mananger import MpiCommunicationManager
from .message import Message
from abc import abstractmethod
from mpi4py import MPI


class MessageManager(Observer):

    def __init__(self, args, node_type, comm=None, rank=0, com_size=0, backend="MPI"):
        self.args = args
        self.com_size = com_size
        self.rank = rank
        self.node_type=node_type
        self.backend = backend
        if backend == "MPI":
            self.com_manager = MpiCommunicationManager(comm, rank, com_size, node_type=node_type)

        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()

    def receive_message(self, msg_type, msg_params) -> None:
        # logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
        #     self.rank, str(msg_type), str(msg_params.get_content())))
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        msg = Message()
        msg.add(Message.MSG_ARG_KEY_TYPE, message.get_type())
        msg.add(Message.MSG_ARG_KEY_SENDER, message.get_sender_id())
        msg.add(Message.MSG_ARG_KEY_RECEIVER, message.get_receiver_id())
        for key, value in message.get_params().items():
            # logging.info("%s == %s" % (key, value))
            msg.add(key, value)
        self.com_manager.send_message(msg)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        if self.backend == "MPI":
            MPI.COMM_WORLD.Abort()