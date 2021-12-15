
## ClientManager
    这个类主要是用来管理当前client什么时候干什么的
    handle_message_semaphore： 当收到从其他client的semaphore信号时进行训练
    handle_message_gradients： 当收到server的梯度时进行反向传播，并确定是否继续训练还是run_eval
    这两个函数是由register_message_receive_handlers注册到MPI里面
