from .log.Log import Log
import time
from mpi4py import MPI

from .client.client import SplitNNClient
from .client.client_manager import ClientManager
from .server.server import SplitNNServer
from .server.server_manager import ServerManager


def SplitNN_init():
    # 初始化MPI
    logging = Log("SplitNN_init")
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    logging.info("process_id: {}, worker_number: {}".format(process_id, worker_number))
    return comm, process_id, worker_number


def SplitNN_distributed(process_id, worker_number, device, comm, client_model,
                        server_model, train_data_num, train_data_global, test_data_global,
                        local_data_num, train_data_local, test_data_local, args):
    logging = Log("SplitNN_distributed")
    server_rank = 0
    if process_id == server_rank:
        logging.info("process_id == server_rank : {}".format(process_id))
        init_server(comm, server_model, process_id, worker_number, device, args)
    else:
        logging.info("process_id == client_rank : {}".format(process_id))
        init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                    process_id, server_rank, args.epochs, device, args)


def init_server(comm, server_model, process_id, worker_number, device, args):
    logging = Log("init_server")
    # arg_dict = {"comm": comm, "server_model": server_model, "max_rank": worker_number - 1,
    #             "rank": process_id, "device": device, "log_step": 50, "args": args}
    args["comm"] = comm
    args["rank"]: process_id
    args["device"] = "cpu"
    args["log_step"] = 50

    server = SplitNNServer(args)
    server_manager = ServerManager(args, server)
    logging.info("Server run begin")
    server_manager.run()
    logging.info("Server run end")


def init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                process_id, server_rank, epochs, device, args):
    logging = Log("init_client")
    # arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
    #             "model": client_model, "rank": process_id, "server_rank": server_rank,
    #             "max_rank": worker_number - 1, "epochs": epochs, "device": device, "args": args}

    args["comm"] = comm
    args["rank"]: process_id
    args["trainloader"] = train_data_local
    args["testloader"] = test_data_local
    args["client_model"] = client_model
    args["server_rank"] = server_rank
    args["max_rank"] = worker_number - 1
    args["epochs"] = epochs
    args["device"] = device

    client = SplitNNClient(args)
    client_manager = ClientManager(args, client)
    logging.info("Client run begin")
    client_manager.run()
