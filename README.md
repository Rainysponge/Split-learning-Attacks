# Split-learning-Attacks

## 文件结构
### /core
  #### &nbsp;&nbsp;/dataset …… 数据集管理类
  ##### &nbsp;&nbsp;&nbsp;&nbsp;/cifar10 cifar10数据集样例
  ##### &nbsp;&nbsp;&nbsp;&nbsp;/partition 分割数据
  &nbsp;&nbsp;&nbsp;&nbsp; partitionUtils.py 分割方法样例  
  &nbsp;&nbsp;&nbsp;&nbsp; partitionFactory.py 分割方法工厂
  #### &nbsp;&nbsp;/log …… 日志管理
  &nbsp;&nbsp;&nbsp;&nbsp; baseLog.py 日志基类  
  &nbsp;&nbsp;&nbsp;&nbsp; Log.py 日志样例
  #### &nbsp;&nbsp;/model …… 模型管理(感觉可以直接用torch)
  #### &nbsp;&nbsp;/optimizer …… 优化器(感觉可以直接用torch)
  #### &nbsp;&nbsp;/client
  #### &nbsp;&nbsp;/communication
### /Parse …… 解析器
  #### &nbsp;&nbsp;&nbsp;&nbsp;/JSON
  #### &nbsp;&nbsp;&nbsp;&nbsp;/YAML …… YAML解析器
   &nbsp;&nbsp;&nbsp;&nbsp;/abstractParse.py …… 解析器基类  
   &nbsp;&nbsp;&nbsp;&nbsp;/abstractParseFactory …… 解析器工厂基类
   
   卷 我的区 的文件夹 PATH 列表
卷序列号为 04C3-25B8
D:.
│  1.txt
│  
└─Split-learning-Attacks
    │  
    ├─communication
    │  │  base_com_manager.py
    │  │  message.py
    │  │  msg_manager.py
    │  │  observer.py
    │  │  __init__.py
    │  │  
    │  ├─mpi
    │  │  │  mpi_com_mananger.py
    │  │  │  mpi_receive_thread.py
    │  │  │  mpi_send_thread.py
    │  │  │  __init__.py
    │          
    └─SLFrame
        │  config.yaml
        │  dataTest.py
        │  log.txt
        │  Test.py
        │  
        │          
        ├─core
        │  │  splitApi.py
        │  │  splitApiO.py
        │  │  
        │  │          
        │  ├─communication
        │  │  │  base_com_manager.py
        │  │  │  message.py
        │  │  │  msg_manager.py
        │  │  │  observer.py
        │  │  │  __init__.py
        │  │  │  
        │  │  ├─mpi
        │  │  │  │  mpi_com_mananger.py
        │  │  │  │  mpi_receive_thread.py
        │  │  │  │  mpi_send_thread.py
        │  │  │  │  __init__.py
        │  │  
        │  │          
        │  ├─dataset
        │  │  │  abstractDataset.py
        │  │  │  abstractDatasetFactory.py
        │  │  │  baseDataset.py
        │  │  │  baseDatasetFactory.py
        │  │  │  datasetFactory.py
        │  │  │  
        │  │  ├─controller
        │  │  │  │  adultController.py
        │  │  │  │  cheXpertController.py
        │  │  │  │  cifar10Controller.py
        │  │  │  │  fashionmnistController.py
        │  │  │  │  germanController.py
        │  │  │  │  mnistController.py
        │  │  │  
        │  │  │          
        │  │  ├─dataset
        │  │  │  │  adult.py
        │  │  │  │  cheXpert.py
        │  │  │  │  cifar10.py
        │  │  │  │  fashionmnist.py
        │  │  │  │  german.py
        │  │  │  │  mnist.py
        │  │  │          
        │  │  ├─partition
        │  │  │  │  basePartition.py
        │  │  │  │  basePartitionFactory.py
        │  │  │  │  cifar10Partition.py
        │  │  │  │  partitionFactory.py
        │  │  │  │  partitionUtils.py
        │  │          
        │  ├─log
        │  │  │  abstractLog.py
        │  │  │  baseLog.py
        │  │  │  Log.py
        │  │  
        │  │          
        │  ├─model
        │  │  │  Alex_model.py
        │  │  │  cnn.py
        │  │  │  DenseNet.py
        │  │  │  EffientNet0.py
        │  │  │  modelFactory.py
        │  │  │  models.py
        │  │  │  models_for_U.py
        │  │  │  model_factory.py
        │  │  │  resnet.py
        │  │          
        │  ├─process
        │  │      baseClient.py
        │  │      baseClientManager.py
        │  │      baseServer.py
        │  │      baseServerManageger.py
        │  │      
        │  │          
        │  ├─variants
        │  │  │  variantsFactory.py
        │  │  │  __init__.py
        │  │  │  
        │  │  ├─Asynchronous
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─asyVanilla
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─asyVanilla2
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─comp_model
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─fedavg
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─parallel_U_Shape
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─SGLR
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─SplitFed
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─SplitFed2
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─synchronization
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │          
        │  │  ├─TaskAgnostic
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │  
        │  │  │          
        │  │  ├─TaskAgnostic2
        │  │  │      client.py
        │  │  │      client_manager.py
        │  │  │      message_define.py
        │  │  │      readme.md
        │  │  │      server.py
        │  │  │      server_manager.py
        │  │  │      __init__.py
        │  │  │      
        │  │  ├─Ushaped
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  
        │  │  │  
        │  │  │          
        │  │  ├─vanilla
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │  │  │  │  __init__.py
        │  │  │  │  
        │  │  │  
        │  │  │          
        │  │  ├─vertical
        │  │  │  │  client.py
        │  │  │  │  client_manager.py
        │  │  │  │  message_define.py
        │  │  │  │  readme.md
        │  │  │  │  server.py
        │  │  │  │  server_manager.py
        │            
        │          
        ├─data
        │  ├─adult
        │  │      adult.data
        │  │      adult.test
        │  │      
        │  ├─CheXpert-v1.0-small
        │  ├─cifar10
        │  │      download_cifar10.sh
        │  │      新建 文本文档.txt
        │  │      
        │  ├─fashionmnist
        │  │  └─FashionMNIST
        │  │              
        │  ├─german
        │  │      german.data-numeric
        │  │      
        │  └─mnist
        │      │  新建 文本文档.txt
        │      │  
        │      └─MNIST
        │                  
        ├─model_save
        │      
        └─Parse
            │  abstractParse.py
            │  baseParse.py
            │  parseFactory.py
            │  utlis.py
            │  
            ├─JSON
            │  │  jsonParse.py
            │  
            │          
            ├─YAML
            │  │  yamlParse.py
    
 
                    






