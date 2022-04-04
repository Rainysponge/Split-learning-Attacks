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
        elif self.variantsType == "asy_vanilla":
            from ..variants.asyVanilla2.server import SplitNNServer
            from ..variants.asyVanilla2.client import SplitNNClient
            from ..variants.asyVanilla2.server_manager import ServerManager
            from ..variants.asyVanilla2.client_manager import ClientManager
        elif self.variantsType == "Asynchronous":
            from ..variants.asyVanilla.server import SplitNNServer
            from ..variants.asyVanilla.client import SplitNNClient
            from ..variants.asyVanilla.server_manager import ServerManager
            from ..variants.asyVanilla.client_manager import ClientManager
        elif self.variantsType == "vertical":
            from ..variants.vertical.server import SplitNNServer
            from ..variants.vertical.client import SplitNNClient
            from ..variants.vertical.server_manager import ServerManager
            from ..variants.vertical.client_manager import ClientManager
        elif self.variantsType == "SGLR":
            from ..variants.SGLR.server import SplitNNServer
            from ..variants.SGLR.client import SplitNNClient
            from ..variants.SGLR.server_manager import ServerManager
            from ..variants.SGLR.client_manager import ClientManager
        elif self.variantsType == "exchange":
            from ..variants.exchange.server import SplitNNServer
            from ..variants.exchange.client import SplitNNClient
            from ..variants.exchange.server_manager import ServerManager
            from ..variants.exchange.client_manager import ClientManager
        elif self.variantsType == "exchange2":
            from ..variants.exchange2.server import SplitNNServer
            from ..variants.exchange2.client import SplitNNClient
            from ..variants.exchange2.server_manager import ServerManager
            from ..variants.exchange2.client_manager import ClientManager
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
