import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :]-x[:, :, :h_x-1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:]-x[:, :, :, :w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x**2).mean()


def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data


def deprocess(data):

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1:
        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
    elif NChannels == 3:
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    Unnormalize = transforms.Normalize(
        (-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0, :, :, :]).unsqueeze(0))


def white_box_inversion(target_model, target_acts, target_img, id, save_path, iterN=500, device="cuda", lr=1e-3, eps=1e-3, lambda_TV=1e3, lambda_l2=1.0):
    xGen = torch.zeros(target_img.size(), requires_grad=True,
                       device=device)
    optimizer = optim.Adam(params=[xGen], lr=lr, eps=eps)
    for i in range(iterN):
        optimizer.zero_grad()
        xFeature = target_model(xGen)
        Featureloss = ((xFeature-target_acts)**2).mean()
        TVloss = TV(xGen)
        normLoss = l2loss(xGen)
        totalLoss = Featureloss + lambda_TV * TVloss + lambda_l2 * normLoss
        totalLoss.backward(retain_graph=True)
        optimizer.step()
        print("Iter:{} total_loss:{}".format(i, totalLoss))

    return xGen
