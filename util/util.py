import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist
from torch import nn, cosine_similarity

import random
import torch
from torch.utils.data import DataLoader, Dataset

from util.Amodel import AutoPromptModel
from util.local_training import LogitAdjust, agg_func


def noisify_label(true_label, num_classes=10, noise_type="symmetric"):
    if noise_type == "symmetric":
        label_lst = list(range(num_classes))
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        return (true_label - 1) % num_classes


def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio

    if args.noise_type == 'symmetric' or args.noise_type == 'pairflip':
        y_train_noisy = copy.deepcopy(y_train)

        for user in range(args.num_users):
            data_indices = list(copy.deepcopy(dict_users[user]))

            random.seed(args.seed)
            random.shuffle(data_indices)
            noise_index = int(len(data_indices) * args.noise_rate)
            for d_idx in data_indices[:noise_index]:
                true_label = y_train_noisy[d_idx]
                noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=args.noise_type)
                y_train_noisy[d_idx] = noisy_label

    return (y_train_noisy, gamma_s, real_noise_level)


def global_protos_gen_auto(global_protos, dataloadertemp):
    xx = []
    for key in sorted(global_protos.keys()):
        xx.append(global_protos[key])
    tensor_xx = torch.stack(xx)
    auto_prompt = AutoPromptModel(tensor_xx).cuda()
    global_protos_stack = torch.stack([global_protos[i] for i in global_protos.keys()])

    optimizer = torch.optim.Adam(auto_prompt.parameters(), lr=0.001)
    for epoch in range(50):
        auto_prompt.train()
        total_num = 0
        num1 = 0
        for data_batch in dataloadertemp:
            labels, image_features = data_batch
            labels, image_features = labels.to('cuda'), image_features.to('cuda')

            CoOp_logits, all_prompts = auto_prompt(image_features)
            loss = F.cross_entropy(CoOp_logits, labels)
            features = all_prompts[torch.arange(all_prompts.size(0)), labels]
            loss += 0.1*info_nce_loss(features, global_protos_stack, labels, temperature=0.1)

            pred = CoOp_logits.topk(1, 1, True, True)[1].t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            num1 += float(correct[: 1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            total_num += labels.data.size()[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return  copy.deepcopy(auto_prompt)


def info_nce_loss(features, prototypes, labels, temperature=0.1):
    num_classes = prototypes.shape[0]
    similarity_matrix = torch.matmul(features, prototypes.T) / temperature
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, torch.arange(num_classes, device=labels.device).unsqueeze(0)).float()
    logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    return loss.mean()


def get_loss_auto_prompt(loader, clip_model, auto_prompt, num_sample_idx, global_proto):
    train_features = []
    cache_keys = []
    cache_values = []
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    criterion = nn.CrossEntropyLoss(reduction='none')
    global_protos = torch.stack([global_proto[i] for i in global_proto.keys()])

    with torch.no_grad():
        for ind, (images2, labels) in enumerate(loader):
            clip_images = images2.to('cuda')
            labels = labels.to('cuda')
            labels = labels.long()
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_images)
            image_features = image_features.float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            CoOp_logits, all_prompts = auto_prompt(image_features)
            loss = criterion(CoOp_logits, labels)

            indices = np.where(np.array(loss.cpu()) < 1)[0]
            train_features.append(image_features[indices])
            cache_values.append(labels[indices])
            if ind == 0:
                loss_whole = np.array(loss.cpu())
            else:
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)

        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        from collections import defaultdict
        protos = defaultdict(list)
        yy = torch.cat(cache_values, dim=0)
        for i in range(len(yy)):
            protos[yy[i].item()].append(cache_keys[i].detach().data)

        protos = agg_func(protos)
    return loss_whole, protos, cache_keys, cache_values, 1-len(yy)/num_sample_idx


def test_proto(loader, clip_model, protos, args):
    loss_mse = nn.MSELoss()
    correct1, total_model, correct2 = 0, 0, 0
    with torch.no_grad():
        for ind, (images2, labels) in enumerate(loader):
            clip_images = images2.to('cuda')
            labels = labels.to('cuda')
            labels = labels.long()
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_images)
            image_features = image_features.float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            outputs = float('inf') * torch.ones(labels.shape[0], args.num_classes).to('cuda')
            for i, r in enumerate(image_features):
                for j, pro in protos.items():
                    if type(pro) != type([]):
                        outputs[i, j] = loss_mse(r, pro)
            correct1 += (torch.sum(torch.argmin(outputs, dim=1) == labels)).item()
            total_model += labels.data.size()[0]
    return correct1/total_model

def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_dataloadertemp(uploaded_protos):
    from collections import defaultdict
    agg_protos_label_temp = defaultdict(list)
    for local_protos in uploaded_protos:
        for label in local_protos.keys():
            agg_protos_label_temp[label].append(local_protos[label])
    dataloader_data = []
    for key, value_list in agg_protos_label_temp.items():
        for value in value_list:
            dataloader_data.append((key, value))
    custom_dataset = CustomDataset(dataloader_data)
    dataloadertemp = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    return dataloadertemp


def train_auto_model(auto_prompt,local_cache_keys, local_cache_values, cls_num_list):
    criterion_ce = LogitAdjust(cls_num_list)
    dataloader_data = []
    yy = torch.cat(local_cache_values, dim=0)
    for ind in range(len(yy)):
        dataloader_data.append((yy[ind], local_cache_keys[ind]))
    custom_dataset = CustomDataset(dataloader_data)
    loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(auto_prompt.parameters(), lr=0.002, eps=1e-3)
    best_model = copy.deepcopy(auto_prompt)

    ans = 0.0
    auto_prompt.train()
    for epoch in range(75):
        total_num = 0
        num1 = 0
        for data_batch in loader:
            labels, image_features = data_batch
            labels, image_features = labels.to('cuda'), image_features.to('cuda')
            CoOp_logits, all_prompts = auto_prompt(image_features)
            #loss = F.cross_entropy(CoOp_logits, labels)
            loss = criterion_ce(CoOp_logits, labels)
            pred = CoOp_logits.topk(1, 1, True, True)[1].t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            num1 += float(correct[: 1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            total_num += labels.data.size()[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = num1/total_num
        if acc > ans:
            ans = acc
            best_model = copy.deepcopy(auto_prompt)
            #print(acc)
    return best_model
