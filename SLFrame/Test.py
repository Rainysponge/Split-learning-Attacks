from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, JSON, YAML


log = Log("Test.py")
log.info("Test")

parse = parseFactory(fileType=YAML).factory()
d = {
    "model": [i for i in range(7)],
    "dataset": "cifar10",
    "dataDir": "./data/cifar10"
}
parse.save(d, './config.yaml')
d = parse.load('./config.yaml')
dataset = datasetFactory().factory(parse)
# for key, value in vars(parse).items():
#     print("{}: {}".format(key, value))


# create_obj = compile('obj()', 'create_obj.py', 'eval')
# print(create_obj)


# class Cat():
#     def __init__(self):
#         self.a = 0
#         self.b = ""
#
#     def A(self):
#         print(self.a)
#
#     def B(self):
#         print(self.b)


