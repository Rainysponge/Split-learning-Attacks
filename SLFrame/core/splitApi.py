from .log.Log import Log
import time
from mpi4py import MPI

from .client.client_manager import ClientManager
from .client.client import SplitNNClient
from .server.server import SplitNNServer
from .server.server_manager import ServerManager
from .model.models import LeNetClientNetwork, LeNetServerNetwork


def SplitNN_init(parse):
    # 初始化MPI
    logging = Log("SplitNN_init", parse)
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    logging.info("process_id: {}, worker_number: {}".format(process_id, worker_number))
    parse["comm"] = comm
    parse["process_id"] = process_id
    parse["worker_number"] = worker_number
    parse["max_rank"] = parse["worker_number"] - 1
    return comm, process_id, worker_number


def SplitNN_distributed(process_id, parse):
    logging = Log("SplitNN_distributed", parse)
    server_rank = 0
    if process_id == server_rank:
        logging.info("process_id == server_rank : {}".format(process_id))
        init_server(parse)
    else:
        logging.info("process_id == client_rank : {}".format(process_id))
        init_client(parse)


def init_server(args):
    logging = Log("init_server", args)

    server = SplitNNServer(args)
    server_manager = ServerManager(args, server)
    logging.info("Server run begin")
    server_manager.run()
    logging.info("Server run end")


def init_client(args):
    logging = Log("init_client", args)
    client = SplitNNClient(args)
    client_manager = ClientManager(args, client)
    logging.info("Client run begin")
    client_manager.run()
