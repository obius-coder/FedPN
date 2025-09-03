# python version 3.7.1
# -*- coding: utf-8 -*-
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, dataset_client_clip, cls_num_list=None):
        self.args = args
        self.sample_idx = idxs
        self.cls_num_list = cls_num_list
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train, self.ldr_test, self.CLIP_train = self.train_test(dataset, dataset_client_clip, list(idxs))
        self.criterion_ce = LogitAdjust(self.cls_num_list)

    def updata_clip_train(self, dataset, dataset_client_clip):
        _,_, self.CLIP_train = self.train_test(dataset, dataset_client_clip, list(self.sample_idx))

    def train_test(self, dataset, dataset_client_clip,  idxs):
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False, num_workers=4, pin_memory=True)
        CLIP_train = DataLoader(DatasetSplit(dataset_client_clip, idxs), batch_size=self.args.local_bs, shuffle=False, num_workers=0, pin_memory=True)
        test = DataLoader(dataset, batch_size=128, num_workers=4)
        return train, test, CLIP_train

    def update_weights(self, net, epoch, lr=None):
        net.train()
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels)  in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def getproto(self, clip_model):
        train_features = []
        cache_keys = []
        cache_values = []

        for batch_idx, (images2, labels) in enumerate(self.CLIP_train):
            clip_images, labels = images2.to(self.args.device), labels.to(self.args.device)
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_images)
            image_features = image_features.float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            train_features.append(image_features)
            cache_values.append(labels)

        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        from collections import defaultdict
        protos = defaultdict(list)
        yy = torch.cat(cache_values, dim=0)
        for i in range(len(yy)):
            protos[yy[i].item()].append(cache_keys[i].detach().data)
        self.protos = agg_func(protos)

        return self.protos


class LocalUpdate2(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, epoch):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=1)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


def agg_func(protos):

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target, get_loss=False):
        x_m = x + self.m_list
        if get_loss == True:
            return  F.cross_entropy(x_m, target, weight=self.weight, reduction='none')
        return F.cross_entropy(x_m, target, weight=self.weight)

