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
    logging = Log("SplitNN_init")
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
    # process_id, worker_number, device, comm, client_model,
    # server_model, train_data_num, train_data_global, test_data_global,
    # local_data_num, train_data_local, test_data_local, args):
    logging = Log("SplitNN_distributed")
    server_rank = 0
    if process_id == server_rank:
        logging.info("process_id == server_rank : {}".format(process_id))
        init_server(parse)
    else:
        logging.info("process_id == client_rank : {}".format(process_id))
        init_client(parse)


def init_server(args):
    logging = Log("init_server")
    # arg_dict = {"comm": comm, "model": server_model, "max_rank": worker_number - 1,
    #             "rank": process_id, "device": device, "args": args}

    # args["device"] = "cpu"
    # args["log_step"] = 50

    server = SplitNNServer(args)
    server_manager = ServerManager(args, server)
    logging.info("Server run begin")
    server_manager.run()
    logging.info("Server run end")


def init_client(args):
    # args["comm"] = comm
    #
    # args["trainloader"] = train_data_local
    # args["testloader"] = test_data_local
    # args["client_model"] = client_model
    # args["server_rank"] = server_rank
    # args["max_rank"] = worker_number - 1
    # args["epochs"] = epochs
    # args["device"] = device
    logging = Log("init_client")
    # args["max_rank"] = args["worker_number"] - 1
    client = SplitNNClient(args)
    client_manager = ClientManager(args, client)
    logging.info("Client run begin")
    client_manager.run()
