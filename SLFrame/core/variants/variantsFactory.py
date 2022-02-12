import logging
from SLFrame.core.log.Log import Log

class variantsFactory():

    def __init__(self, variantsType, nodeType, args):
        self.variantsType = variantsType
        self.nodeType = nodeType
        self.args=args
        self.log = Log("parseFactory",args)

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
        elif self.variantsType == "asy_vanilla":
            from ..variants.asyVanilla2.server import SplitNNServer
            from ..variants.asyVanilla2.client import SplitNNClient
            from ..variants.asyVanilla2.server_manager import ServerManager
            from ..variants.asyVanilla2.client_manager import ClientManager
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
            return client,client_manager