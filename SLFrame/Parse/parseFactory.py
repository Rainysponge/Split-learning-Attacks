from .baseParse import abstractParseFactory
from .JSON.jsonParse import jsonParse
from .YAML.yamlParse import yamlParse

JSON = 1
YAML = 2


class parseFactory(abstractParseFactory):
    """
    样例
    """

    def __init__(self, fileType):
        self.fileType = fileType
       # self.log = Log("parseFactory")

    def factory(self):
        if self.fileType == JSON:
            return jsonParse()
        if self.fileType == YAML:
            return yamlParse()
