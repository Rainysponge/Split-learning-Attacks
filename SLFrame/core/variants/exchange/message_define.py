class MyMessage(object):
    """
        message type definition
    """
    # SplitServer to client
    MSG_TYPE_S2C_GRADS = 1

    # client to SplitServer
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5

    # client to client
    MSG_TYPE_C2C_SEMAPHORE = 6

    # client to FedServer
    MSG_TYPE_C2S_SEND_MODEL = 7
    MSG_TYPE_C2S_SEND_WEIGHTS = 8

    # FedServer to client
    MSG_TYPE_S2C_SEND_MODELS = 9

    # message base information definition
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    # message payload keywords definition
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_AGR_KEY_RESULT = "result"
    MSG_ARG_KEY_MODEL = "model"
    MSG_ARG_KEY_WEIGHTS = "weights"
    MSG_ARG_KEY_MODELS = "models"
