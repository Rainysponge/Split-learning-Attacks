## config

config文件参数

### dataDir

数据存放的文件夹

### dataset

数据集(mnist/ adult/ cifar)

### device
cpu/gpu

### partition_method
数据分割方式

**homo**: 普通随机IID

**base_on_class**: 每一个客户端只能被分配到指定label的数据

**..**:..

### variants_type

SL的范式

**vanilla**: 每个客户端轮流进行学习

**asy_vanilla**: 每个客户端同时进行学习, 服务端按时间顺序响应客户端的请求

**Ushaped**: 每个客户端轮流进行保护label的学习

**vertical**: 把多个客户端的隐藏层垂直切割后再合并起来学习

### server_rank

指定服务端的rank, 默认为0

### config
**dataDir**: 存放数据集的文件夹
**dataset**: 要使用的数据集
**download**: 是否需要下载数据集 针对pytorch提供的数据集如mnist
**batch_size**: batch大小
**log_step**: 每几个epochs记录一次日志
**lr**: 优化器参数
**partition_method**: 数据集拆分方法
**variants_type**: 范式


