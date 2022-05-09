class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_MODEL = 10
    MSG_TYPE_S2C_ACTS = 8
    MSG_TYPE_S2C_SEMAPHORE = 6
    MSG_TYPE_S2C_READY_TO_GET_MODEL = 14

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_SEND_GRADS = 7
    MSG_TYPE_C2S_MODEL = 11
    MSG_TYPE_C2S_NEXT_BATCH = 13
    MSG_TYPE_C2S_READY_TO_SEND_MODEL = 15
    # 优先队列 wait type

    # c 2 c
    MSG_TYPE_C2C_SEMAPHORE = 12


    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_AGR_KEY_RESULT = "result"

    MSG_TYPE_TEST_C2C = 9

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"

    # MSG_AGR_KEY_MODEL = "model"
    MSG_AGR_HEAD_MODEL = "head_model"
    MSG_AGR_TAIL_MODEL = "tail_model"
    MSG_AGR_KEY_SAMPLE_NUM = "sample_num"
    # MSG_AGR_KEY_MODEL
    MSG_ARG_KEY_CLIENT_NUM = "client_num"
    MSG_ARG_KEY_CUR_DATASET_IDX = "dataset_cur"

