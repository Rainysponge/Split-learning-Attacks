from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, JSON, YAML

# log = Log("Test.py")
# log.info("Test")
#
parse = parseFactory(fileType=YAML).factory()
d = {
    "model": [i for i in range(7)],
    "dataset": "cifar10",
    "dataDir": "./data/cifar10",
    'download': True,
    'partition_method': 'homo'

}
parse.save(d, './config.yaml')
d = parse.load('./config.yaml')
dataset = datasetFactory().factory(parse)  # loader data and partition method
print(dataset)
dataset.partition_data()




