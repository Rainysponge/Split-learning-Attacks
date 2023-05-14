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
  


## 如何使用
在下载整个文件后 需要配置相关环境，安装mpi和下列库
```
yaml
py4mpi
pytorch
scikit-learn
pandas
numpy
pillow
matplotlib
```

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
也可以继承process中的相关类已使用一些通用的方法。


## 与fedml对比
FedML和本框架有许多相似之处，如都采用了Mpi作为通讯方式，对于数据集都才用pytorch.dataset中的dataloader作为数据划分结果等，但仍有不少的区别。  
FedML支持的是联邦学习，绝大多数情况下，训练在一个端中就已经完成了，不会有训练数据和梯度在端与端中的交互，而本框架需要考虑这个问题。*同时FedML在使用的时候一般是脚本文件配合命令行来完成，且仅支持linux环境，因为要用到sh文件作为输入，而本框架因为使用yaml或者json文件为配置输入方式则可以适配win和linux两种环境*（现在的fedml也支持配置文件输入）。由于本框架采用了工厂方法设计模式进行设计，所以在代码修改和拓展上会比FedML更为方便，虽然一般来说工厂方法会带来代码的复杂性，但是对于FedML而言因为没有使用该方法，所以主函数会出现大量的if-else语句对命令行输入做判断，在另一个程度上提高了学习门槛。  
由于FedML没有使用工厂方法，所以在使用时会出现大量重复的if-else语句，一个主函数往往由三四百行且都是重复的代码，过分冗杂。同时因为没有相应的范式工厂，每一个范式都需要一重新一个范例代码，使整个文件看起来十分复杂，在我刚开始看源码时经常不知道如何入手。简单罗列本框架与FedML的不同如下：  
(1)	FedML仅适用于linux环境，而本框架支持linux和windows。  
*(2)	FedML使用命令行和sh文件作为输入，且不使用方法工厂，使得部分代码重复度较高，不利于拓展；而本框架才用方法工厂，使得代码更具有拓展性。*  
(3)	本框架适用于拆分学习而FedML是针对联邦学习开发的框架。  
*(4)	本框架采用yaml和json作为配置文件，相较于sh文件更好编辑。*  
(5)	FedML不仅支持MPI通讯，还支持其他通讯协议，而本框架仅支持MPI通讯。  
(6)	FedML支持多卡运算和移动端配置，而本框架暂不支持。  



