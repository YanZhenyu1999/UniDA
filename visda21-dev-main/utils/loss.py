import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


def ova_loss(out_open, label):
# def ova_loss(out_open, label, out_s):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.range(0, out_open.size(0)-1).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    # open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] + 1e-8) * label_n, 1)[0]) # 0维bs，对第1维取最大值，返回value index取[0]value

    # postive = out_open[:, 1, :] * label_p
    # consistency_loss = torch.nn.function  al.mse_loss(postive, out_s)

    # topk
    value, index = torch.topk(-torch.log(out_open[:, 0, :] + 1e-8) * label_n, k=2, dim=1)    # 最高k个，对第一维，0是batch
    value = torch.mean(value, dim=1)    # 对k个求均值
    open_loss_neg = torch.mean(value)
    return open_loss_pos, open_loss_neg#, consistency_loss


# def open_entropy(out_open):
def open_entropy(out_open, out_t):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    # target consistency
    out_t = F.softmax(out_t)
    out_t_value, out_t_index = torch.max(out_t, 1)
    # print('out_t_value ', out_t_value.size())
    label_range = torch.range(0, out_open.size(0) - 1).long()
    out_open_pre = out_open[label_range, 1, out_t_index]
    # print('out_open_pre ', out_open_pre.size())
    consist_loss = torch.nn.functional.mse_loss(out_t_value, out_open_pre)
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open, consist_loss

