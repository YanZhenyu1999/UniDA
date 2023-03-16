from __future__ import print_function
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models
from eval import test
import argparse

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/image_to_objectnet.yaml',
                    help='/path/to/config/file')

parser.add_argument('--source_data', type=str,
                    default='/home/yzy/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./val_filelists/objectnet_filelist.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='ovanet',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=False, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/ova_model",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.1,
                    help='weight factor for adaptation')
parser.add_argument("--cds",
                    default=False, action='store_true')
parser.add_argument('--cds_path', type=str,
                    default="./checkpoints/CDS_Imagenet1k_imagenet1k.txt_objectnet_c_r_o.txt_epoch_0_15999.t7",
                    help='cross domain sel-supervised pretraining')

args = parser.parse_args()

log_file = open('train.log', 'w')

config_file = args.config
conf = yaml.safe_load(open(config_file))
save_config = yaml.safe_load(open(config_file))
conf = easydict.EasyDict(conf)

if args.gpu_devices == None:
    gpu_devices = '0'
else:
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

source_loader, target_loader, \
test_loader, target_folder = get_dataloaders(inputs)

logname = log_set(inputs)

G, C1, C2, D, opt_g, opt_c, opt_d, \
param_lr_g, param_lr_c, param_lr_d = get_models(inputs)
ndata = target_folder.__len__()

target_loader.dataset.labels[target_loader.dataset.labels > 1000] = 1000
test_loader.dataset.labels[test_loader.dataset.labels > 1000] = 1000


def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    acc = []
    hscore = []
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_d, opt_d, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        opt_d.zero_grad()
        C2.module.weight_norm()

        ## Source loss calculation
        feat_s = G(img_s)
        out_s = C1(feat_s)
        out_open = C2(feat_s)
        ## source classification loss
        loss_s = criterion(out_s, label_s)
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = ova_loss(out_open, label_s)
        # open_loss_pos, open_loss_neg, consistency_loss = ova_loss(out_open, label_s, out_s)
        ## b x 2 x C
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all = loss_s + loss_open  # + consistency_loss
        log_string = 'Train {}/{} \t ' \
                     'Loss Source: {:.4f} ' \
                     'Loss Open: {:.4f} ' \
                     'Loss Open Source Positive: {:.4f} ' \
                     'Loss Open Source Negative: {:.4f} ' \
            # 'Loss consistency: {:.4f}'
        log_values = [step, conf.train.min_step,
                      loss_s.item(), loss_open.item(),
                      open_loss_pos.item(), open_loss_neg.item()  # , consistency_loss.item()
                      ]

        # Target loss calculation
        feat_t = G(img_t)
        out_t = C1(feat_t)
        out_open_t = C2(feat_t)
        out_open_t = out_open_t.view(img_t.size(0), 2, -1)

        # target open loss, consist loss
        if not args.no_adapt:
            ent_open, consist_loss = open_entropy(out_open_t, out_t)
            all += args.multi * ent_open # + consist_loss
            log_values.append(ent_open.item())
            log_values.append(consist_loss.item())
            log_string += "Loss Open Target: {:.6f} " \
                          "Loss consistency: {:.4f} "

        # adversarial loss
        # pred = out_s.data.max(1)[1]
        # out_open_s = F.softmax(out_open, 1)
        # tmp_range_s = torch.range(0, out_s.size(0) - 1).long().cuda()
        # ws = out_open_s[tmp_range_s, 1, pred]
        #
        # pred = out_t.data.max(1)[1]
        # out_open_t = F.softmax(out_open_t, 1)
        # tmp_range_t = torch.range(0, out_t.size(0) - 1).long().cuda()
        # wt = out_open_t[tmp_range_t, 1, pred]

        d_prob_s = D(feat_s)
        d_prob_t = D(feat_t)
        tmp = (nn.BCEWithLogitsLoss(reduction='none')(d_prob_s, torch.ones_like(d_prob_s))).squeeze(1)
        # print('444', tmp.size())
        adv_loss = torch.mean(tmp, dim=0)
        # print('555', wt.size())
        tmp = (nn.BCEWithLogitsLoss(reduction='none')(d_prob_t, torch.zeros_like(d_prob_t))).squeeze(1)
        # print('666', tmp.size())
        adv_loss += torch.mean(tmp, dim=0)
        # print('777', adv_loss.size())
        all += adv_loss
        log_values.append(adv_loss.item())
        log_string += "Loss adversarial: {:.4f} "

        with amp.scale_loss(all, [opt_g, opt_c, opt_d]) as scaled_loss:
            scaled_loss.backward()
        # for parameters in G.module.model.post_network.parameters():
        #     print(parameters.grad)
        opt_g.step()
        opt_c.step()
        opt_d.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        opt_d.zero_grad()
        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))
            log_file.write(log_string.format(*log_values) + '\n')
        if step % conf.test.test_interval == 0:
            acc_o, h_score = test(step, test_loader, logname, n_share, G,
                                  [C1, C2], open=open)
            print("acc all %s h_score %s " % (acc_o, h_score))
            log_file.write('\n' + "acc all %s h_score %s " % (acc_o, h_score) + '\n')
            acc.append(acc_o)
            hscore.append(h_score)
            G.train()
            C1.train()
            if args.save_model:
                save_path = "%s_%s.pth" % (args.save_path, step)
                save_model(G, C1, C2, D, save_path)

    log_file.write('\n' + "best acc %s h_score %s " % (max(acc), hscore[acc.index(max(acc))]) + '\n')


train()
log_file.close()
