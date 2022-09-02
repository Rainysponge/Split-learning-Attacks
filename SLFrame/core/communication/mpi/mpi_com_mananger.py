import sys
import queue
import time
from typing import List
import logging
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from .mpi_receive_thread import MPIReceiveThread
from .mpi_send_thread import MPISendThread
from ..observer import Observer


class MpiCommunicationManager(BaseCommunicationManager):
    def __init__(self, comm, rank, size, node_type="client"):
        self.comm = comm
        self.rank = rank
        self.size = size

        self._observers: List[Observer] = []
        self.send_thread = None
        self.receive_thread = None
        self.collective_thread = None


        if node_type == "client":
            self.q_sender, self.q_receiver = self.init_client_communication()
        elif node_type == "server":
            self.q_sender, self.q_receiver = self.init_server_communication()

        self.is_running = True
        self.reset_analysis_data()

    def init_server_communication(self):
        """
            创建并返回发送接收队列，利用线程来给队列进行更新
        """
        server_send_queue = queue.Queue()
        self.send_thread = MPISendThread(self.comm, self.rank, self.size, "ServerSendThread", server_send_queue)
        self.send_thread.start()

        server_receive_queue = queue.Queue()
        self.receive_thread = MPIReceiveThread(self.comm, self.rank, self.size, "ServerReceiveThread",
                                                      server_receive_queue)
        self.receive_thread.start()
        return server_send_queue, server_receive_queue

    def init_client_communication(self):
        # SEND
        client_send_queue = queue.Queue()
        self.send_thread = MPISendThread(self.comm, self.rank, self.size, "ClientSendThread", client_send_queue)
        self.send_thread.start()

        # RECEIVE
        client_receive_queue = queue.Queue()
        self.receive_thread = MPIReceiveThread(self.comm, self.rank, self.size, "ClientReceiveThread",
                                                      client_receive_queue)
        self.receive_thread.start()
        return client_send_queue, client_receive_queue

    def reset_analysis_data(self):
        self.send_thread.tmp_send_size = 0
        self.receive_thread.tmp_receive_size = 0

    def send_message(self, msg: Message, priority=100):
        msg.add_params(Message.MSG_ARG_KEY_RECEIVE_PRIORITY,priority)
        self.q_sender.put(msg)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self.is_running = True
        while self.is_running:
            if not self.q_receiver.empty():
                msg_params = self.q_receiver.get()
                self.notify(msg_params)

            time.sleep(0.01)

    def wait_for_message(self):
        while True:
            if self.q_receiver.qsize() > 0:
                msg_params = self.q_receiver.get()
                return msg_params
            else:
                time.sleep(0.01)

    def stop_receive_message(self):
        self.is_running = False
        self.__stop_thread(self.send_thread)
        self.__stop_thread(self.receive_thread)
        self.__stop_thread(self.collective_thread)

    def notify(self, msg_params):
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def __stop_thread(self, thread):
        if thread:
            thread.raise_exception()
            thread.join()
