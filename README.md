# Split-learning-Attacks


## 文件结构
   
SLFrame  
      │  config.yaml  
      │  dataTest.py  
      │  log.txt  
      │  Test.py 测试用主程序  
      │            
      ├─core  
      │  │  splitApi.py  
      │  │  splitApiO.py  
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
      │  ├─dataset …… 数据集管理类  
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
      │  │  ├─partition  分割数据  
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
      │  ├─variants  
      │  │  │  variantsFactory.py  
      │  │  │    
      │  │  ├─Asynchronous  
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
      │  │  ├─comp_model   
      │  │  │              
      │  │  ├─fedavg  
      │  │  │  
      │  │  │          
      │  │  ├─parallel_U_Shape   
      │  │  │       
      │  │  ├─SGLR  
      │  │  │         
      │  │  ├─SplitFed  
      │  │  │            
      │  │  ├─SplitFed2  
      │  │  │           
      │  │  ├─synchronization  
      │  │  │           
      │  │  ├─TaskAgnostic  
      │  │  │                 
      │  │  ├─Ushaped  
      │  │  │              
      │  │  ├─vanilla  
      │  │  │              
      │  │  ├─vertical  
      │              
      │            
      ├─data  
      │  ├─adult  
      │  │      adult.data  
      │  │      adult.test  
      │  │      
      │  ├─CheXpert-v1.0-small  
      │  ├─cifar10  
      │  │        
      │  ├─fashionmnist  
      │  │  └─FashionMNIST  
      │  │                
      │  ├─german  
      │  │      german.data-numeric  
      │  │        
      │  └─mnist  
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
          ├─YAML …… YAML解析器 
          │  │  yamlParse.py  
  
在下载整个文件后 // 配置环境
## 如何使用
如要使用已有的范式只需要配置config.yaml文件  
这里用psl举例，配置文件如下：  

```
dataDir: ./data/
dataset: mnist
device: cpu
download: true
batch_size: 128
epochs: 15
log_step: 20
lr: 0.01
max_rank: 2
partition_method: homo
variants_type: psl
server_rank: 0


```
其中varieants_type参数表示的是范式类型； partition_method指代数据集拆分方法；dataset表示所用数据集；epoch和batch_size分别指要训练了多少轮和每轮的batch大小。  
主函数举例如下  
```
if __name__ == '__main__':
    """
        尽量让Test.py是可以不需要做任何其他操作直接运行的
    """
    args = parseFactory(fileType=YAML).factory()
    client_model = LeNetClientNetwork()
    
    server_model = LeNetServerNetwork()
    args.load('./config.yaml')

    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id  # 设置当前process_id

    args["client_model"] = client_model
    args["server_model"] = server_model
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args["device"] = device

    dataset = datasetFactory(args).factory()  # loader data and partition method

    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id) 

    SplitNN_distributed(process_id, args)
```
需要在主函数中声明的是所用的模型即
```
    args["client_model"] = client_model
    args["server_model"] = server_model
```
运行则通过在命令行输入
`mpiexec -np N python filename`
N 表示进程个数，相当于客户端数加1，'filename'表示主函数所在文件名。  

## 自定义范式
可以模仿其他范式的书写方法进行撰写，必须要有的文件为
```
client.py  // 定义客户端的相关参数和训练方法
client_manager.py  // 定义客户端在通讯时的相关方法
message_define.py  // 定义通讯所需的相关参数
server.py  // 定义服务端的相关参数和训练方法
server_manager.py  // 定义服务端在通讯时的相关方法
```
也可以继承process中的相关类已使用一些通用的方法




