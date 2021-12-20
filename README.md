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





