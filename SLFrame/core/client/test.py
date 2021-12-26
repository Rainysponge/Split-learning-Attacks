import sys
from mpi4py import MPI

sys.path.extend("../../../")
from SLFrame.core.client.client_manager import ClientManager
from SLFrame.core.client.client import SplitNNClient
from SLFrame.core.server.server import SplitNNServer
from SLFrame.core.server.server_manager import ServerManager
from SLFrame.core.model.models import LeNetClientNetwork, LeNetServerNetwork


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    com_size = comm.Get_size()
    args = {}
    args["comm"] = comm
    args["rank"] = process_id
    args["max_rank"] = com_size - 1

    """
    把一些参数放进args, 比如把模型切割好放到args["model"]里之类的
    """
    client_model = LeNetClientNetwork()
    server_model = LeNetServerNetwork()
    args["model"] = client_model
    # args["trainloader"] = train_data_local
    # args["testloader"] = test_data_local
    args["rank"] = 1
    args["max_rank"] = 2
    args["lr"] = 0.01
    args["epochs"] = 2
    args["server_rank"] = 1
    args['device'] = 'cpu'
    args["max_rank"] = 4

    args["log_step"] = 50
    if process_id == 0:
        server = SplitNNServer(args)
        server_manager = ServerManager(args, server)
    else:
        client = SplitNNClient(args)
        client_manager = ServerManager(args, client)
