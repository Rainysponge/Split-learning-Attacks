## 过程
1. Server和Client2一开始处于等待状态，Server将active_node设置为1，表示当前正在和Client1交互
2. active_node进行一次forward_pass
3. active_node发送acts给Server，并开始等待
4. Server收到acts，进行训练，再吧acts2发送回active_node
5. active_node收到acts2，进行训练：
   1. 若当前是训练阶段，则将损失的梯度反向传播，并且传播到最后判断当前是否是最后一个训练数据，若是则启动验证阶段。最后重复2。
   2. 若当前是验证阶段，不返回梯度，只记录准确率等信息，若当前是最后一个验证数据，并且（如果当前是最后一个epoch的最后一个节点就结束，否则通知服务器将active_node改成下一个节点，并通知下一个节点开始训练，自己进入等待）。最后重复2
