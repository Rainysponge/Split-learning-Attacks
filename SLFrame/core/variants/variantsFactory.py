import logging
from core.log.Log import Log

class variantsFactory():

    def __init__(self, variantsType, nodeType, args):
        self.variantsType = variantsType
        self.nodeType = nodeType
        self.args = args
        self.log = Log("parseFactory", args)

    def factory(self):
        if self.variantsType == "vanilla":
            from ..variants.vanilla.server import SplitNNServer
            from ..variants.vanilla.client import SplitNNClient
            from ..variants.vanilla.server_manager import ServerManager
            from ..variants.vanilla.client_manager import ClientManager
        elif self.variantsType == "Ushaped":
            from ..variants.Ushaped.server import SplitNNServer
            from ..variants.Ushaped.client import SplitNNClient
            from ..variants.Ushaped.server_manager import ServerManager
            from ..variants.Ushaped.client_manager import ClientManager
        elif self.variantsType == "gnn":
            from ..variants.gnn.server import SplitNNServer
            from ..variants.gnn.client import SplitNNClient
            from ..variants.gnn.server_manager import ServerManager
            from ..variants.gnn.client_manager import ClientManager
        elif self.variantsType == "parallel_U_Shape":
            from ..variants.parallel_U_Shape.server import SplitNNServer
            from ..variants.parallel_U_Shape.client import SplitNNClient
            from ..variants.parallel_U_Shape.server_manager import ServerManager
            from ..variants.parallel_U_Shape.client_manager import ClientManager
        elif self.variantsType == "asy_vanilla":
            from ..variants.asyVanilla2.server import SplitNNServer
            from ..variants.asyVanilla2.client import SplitNNClient
            from ..variants.asyVanilla2.server_manager import ServerManager
            from ..variants.asyVanilla2.client_manager import ClientManager
        elif self.variantsType == "asy_vanilla2":
            from ..variants.asyVanilla.server import SplitNNServer
            from ..variants.asyVanilla.client import SplitNNClient
            from ..variants.asyVanilla.server_manager import ServerManager
            from ..variants.asyVanilla.client_manager import ClientManager
        # elif self.variantsType == "Asynchronous":
        #     from ..variants.asyVanilla.server import SplitNNServer
        #     from ..variants.asyVanilla.client import SplitNNClient
        #     from ..variants.asyVanilla.server_manager import ServerManager
        #     from ..variants.asyVanilla.client_manager import ClientManager
        elif self.variantsType == "vertical":
            from ..variants.vertical.server import SplitNNServer
            from ..variants.vertical.client import SplitNNClient
            from ..variants.vertical.server_manager import ServerManager
            from ..variants.vertical.client_manager import ClientManager
        elif self.variantsType == "Asynchronous":
            from ..variants.Asynchronous.server import SplitNNServer
            from ..variants.Asynchronous.client import SplitNNClient
            from ..variants.Asynchronous.server_manager import ServerManager
            from ..variants.Asynchronous.client_manager import ClientManager
        elif self.variantsType == "SGLR":
            from ..variants.SGLR.server import SplitNNServer
            from ..variants.SGLR.client import SplitNNClient
            from ..variants.SGLR.server_manager import ServerManager
            from ..variants.SGLR.client_manager import ClientManager
        elif self.variantsType == "SplitFed":
            from ..variants.SplitFed.server import SplitNNServer
            from ..variants.SplitFed.client import SplitNNClient
            from ..variants.SplitFed.server_manager import ServerManager
            from ..variants.SplitFed.client_manager import ClientManager
        elif self.variantsType == "SplitFed2":
            from ..variants.SplitFed2.server import SplitNNServer
            from ..variants.SplitFed2.client import SplitNNClient
            from ..variants.SplitFed2.server_manager import ServerManager
            from ..variants.SplitFed2.client_manager import ClientManager
        elif self.variantsType == "synchronization":
            from ..variants.synchronization.server import SplitNNServer
            from ..variants.synchronization.client import SplitNNClient
            from ..variants.synchronization.server_manager import ServerManager
            from ..variants.synchronization.client_manager import ClientManager
        elif self.variantsType == "TaskAgnostic":
            from ..variants.TaskAgnostic.server import SplitNNServer
            from ..variants.TaskAgnostic.client import SplitNNClient
            from ..variants.TaskAgnostic.server_manager import ServerManager
            from ..variants.TaskAgnostic.client_manager import ClientManager
        elif self.variantsType == "comp_model":
            from ..variants.comp_model.server import SplitNNServer
            from ..variants.comp_model.client import SplitNNClient
            from ..variants.comp_model.server_manager import ServerManager
            from ..variants.comp_model.client_manager import ClientManager
        else:
            logging.warning("variants_type: default as vanilla")
            from ..variants.vanilla.server import SplitNNServer
            from ..variants.vanilla.client import SplitNNClient
            from ..variants.vanilla.server_manager import ServerManager
            from ..variants.vanilla.client_manager import ClientManager

        if self.nodeType == "server":
            server = SplitNNServer(self.args)
            server_manager = ServerManager(self.args, server)
            return server, server_manager
        else:
            client = SplitNNClient(self.args)
            client_manager = ClientManager(self.args, client)
            return client, client_manager
