import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from apex import amp, optimizers
from data_loader.get_loader import get_loader, get_loader_label
from .utils import get_model_mme
from models.basenet import ResClassifier_MME, AdversarialNetwork


def get_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)



def get_dataloaders_label(source_data, target_data, target_data_label, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_label(source_data, target_data, target_data_label,
                            evaluation_data, data_transforms,
                            batch_size=conf.data.dataloader.batch_size,
                            return_id=True,
                            balanced=conf.data.dataloader.class_balance)

def get_models(kwargs):
    net = kwargs["network"]
    cds = kwargs["cds"]
    cds_path = kwargs["cds_path"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    G, dim = get_model_mme(net, num_class=num_class, cds=cds, cds_path=cds_path)
    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)
    C2 = ResClassifier_MME(num_classes=2 * num_class,
                           norm=False, input_size=dim)

    D = AdversarialNetwork(in_feature=dim)
    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    D.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    opt_g = optim.SGD(G.parameters(), lr=0.01, momentum=conf.train.sgd_momentum,
                      weight_decay=0.0005, nesterov=True)
    if net == "volod3":
        opt_g = optim.AdamW(G.parameters(), lr=1.0, weight_decay=0.05)
    opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                       momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)
    # opt_c = optim.SGD(C2.parameters(), lr=1.0,
    #                    momentum=conf.train.sgd_momentum, weight_decay=0.0005,
    #                    nesterov=True)
    opt_d = optim.SGD(D.parameters(), lr=1.0, weight_decay=conf.train.weight_decay,
                  momentum=conf.train.sgd_momentum, nesterov=True)

    [G, C1, C2, D], [opt_g, opt_c, opt_d] = amp.initialize([G, C1, C2, D],
                                                  [opt_g, opt_c, opt_d],
                                                  opt_level="O1")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    D = nn.DataParallel(D)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_d = []
    for param_group in opt_d.param_groups:
        param_lr_d.append(param_group["lr"])

    return G, C1, C2, D, opt_g, opt_c, opt_d, param_lr_g, param_lr_c, param_lr_d