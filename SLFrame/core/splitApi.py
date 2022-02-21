from .log.Log import Log
import time
from mpi4py import MPI

from .variants.variantsFactory import variantsFactory
from .model.models import LeNetClientNetwork, LeNetServerNetwork


def SplitNN_init(parse):
    # 初始化MPI
    # mpiexec -np 3 python Test.py
    logging = Log("SplitNN_init", parse)
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    logging.info("process_id: {}, worker_number: {}".format(process_id, worker_number))
    parse["comm"] = comm
    parse["process_id"] = process_id
    parse["rank"] = process_id
    parse["worker_number"] = worker_number
    parse["client_number"] = worker_number - 1
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
    server,server_manager = variantsFactory(args["variants_type"], "server", args).factory()
    logging.info("Server run begin {}".format(args["variants_type"]))
    server_manager.run()
    logging.info("Server run end")


def init_client(args):
    logging = Log("init_client", args)
    client, client_manager = variantsFactory(args["variants_type"], "client", args).factory()
    logging.info("Client {} run begin".format(args['rank']))
    client_manager.run()
