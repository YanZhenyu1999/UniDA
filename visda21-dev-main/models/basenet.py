from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from easydl import *

from .volo import *


class VoloBase(nn.Module):
    def __init__(self, option='volo', pret=True, feat=True):
        super(VoloBase, self).__init__()
        self.dim = 512
        if option == 'volod1':
            self.model = volo_d1(pretrianed=pret, return_feat=feat)
            self.dim = 384
        if option == 'volod2':
            self.model = volo_d2(pretrianed=pret, return_feat=feat)
        if option == 'volod3':
            self.model = volo_d3(pretrained=pret, return_feat=feat)
            state_dict = torch.load('./checkpoints/d3_224_85.4.pth.tar')
            self.model.load_state_dict(state_dict, strict=True)
            for param in self.model.parameters():
                param.requires_grad = False
        if option == 'volod4':
            self.model = volo_d4(pretrianed=pret, return_feat=feat)
            self.dim = 768
        if option == 'volod5':
            self.model = volo_d5(pretrianed=pret, return_feat=feat)
            self.dim = 768

    def forward(self, x):
        x = self.model(x)
        return x



class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False, cds=False, cds_path=None):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)

        if cds:
            state_dict = torch.load(cds_path)
            # state_dict = torch.load("./checkpoints/CDS_Imagenet1k_imagenet1k.txt_objectnet_c_r_o.txt_epoch_0_15999.t7")
            # state_dict = torch.load("./checkpoints/CDS_Imagenet1k_imagenet1k.txt_imagenet_c_r_o_filelist.txt_epoch_0_10999.t7")
            # state_dict = torch.load("./checkpoints/CDS_Imagenet1k_imagenet1k.txt_objectnet_filelist.txt_epoch_0_4999.t7")
            pretrained_dict = {k: v for k, v in state_dict['net'].items() if k.find('fc') == -1}
            model_ft.load_state_dict(pretrained_dict, strict=False)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x


class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False, top=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        self.top = top

        if option =='vgg11_bn':
            vgg16=models.vgg11_bn(pretrained=pret)
        elif option == 'vgg11':
            vgg16 = models.vgg11(pretrained=pret)
        elif option == 'vgg13':
            vgg16 = models.vgg13(pretrained=pret)
        elif option == 'vgg13_bn':
            vgg16 = models.vgg13_bn(pretrained=pret)
        elif option == "vgg16":
            vgg16 = models.vgg16(pretrained=pret)
        elif option == "vgg16_bn":
            vgg16 = models.vgg16_bn(pretrained=pret)
        elif option == "vgg19":
            vgg16 = models.vgg19(pretrained=pret)
        elif option == "vgg19_bn":
            vgg16 = models.vgg19_bn(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        if self.top:
            self.vgg = vgg16

    def forward(self, x, source=True,target=False):
        if self.top:
            x = self.vgg(x)
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), 7 * 7 * 512)
            x = self.classifier(x)
            return x


class ResClassifier_MME1(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME1, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=True)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=True)
        self.norm = norm
        self.tmp = temp
        state_dict = torch.load('./checkpoints/d3_224_85.4.pth.tar')
        # print(state_dict.keys())
        keys_fc = {}
        for k, v in state_dict.items():
            if k.startswith('head'):
                k_fc = k.replace('head', 'fc')
                keys_fc[k_fc] = v
        self.load_state_dict(keys_fc)
        for params in self.fc.parameters():
            params.requires_grad = False

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)



class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)



class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y