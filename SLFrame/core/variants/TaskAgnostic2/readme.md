## 过程
1. server处于等待状态, 其他所有client设置为训练状态
2. client开始跑forward_pass, 并且不管其他客户端状态, 直接向服务端发送acts和label和当前状态(训练/验证).
3. server收到client发来的acts和label(可能同时收到多个, 会按照收到的顺序依次处理), 记录发送此次acts的node为active_node
4. server跑一次forward_pass, 根据active_node的状态来确定:
   1. 若为训练阶段, 则反向传播, 把梯度和运行结果返回给client
   2. 若为验证阶段, 则只返回运行结果
5. client收到server发回的消息:
   1. 若为训练阶段, 用运行结果计算正确率, 用梯度给自己反向传播更新, 训练数据用完了就转换为验证阶段重复2
   2. 若为验证阶段, 用运行结果计算正确率, 验证集用完了判断是否继续训练, 是的话就转换为训练状态重复2, 否则结束