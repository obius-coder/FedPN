
import time

import matplotlib


matplotlib.use('Agg')

import copy
import numpy as np
import random
import torch
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest, LocalUpdate2
from util.fedavg import FedAvg
from util.util import add_noise, proto_aggregation, \
    test_proto, get_dataloadertemp, global_protos_gen_auto, get_loss_auto_prompt, train_auto_model

from util.dataset import get_dataset, get_dataset_clip
from model.build_model import build_model
import clip

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

if __name__ == '__main__':
    args = args_parser()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    dataset_train, dataset_test, dict_users = get_dataset(args)
    # ---------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_real_label = copy.deepcopy(dataset_train.targets)

    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    netglob = build_model(args)
    net_local = build_model(args)
    clip_model, preprocess = clip.load('ViT-B/16', 'cuda')
    clip_model.eval()
    clip_data_local_training = get_dataset_clip(args, preprocess)
    clip_data_local_training.targets = y_train_noisy
    clients = []
    for idx in range(args.num_users):
        sample_idx = np.array(list(dict_users[idx]))
        class_data = [[] for i in range(args.num_classes)]
        idx_labels = dataset_train.targets[sample_idx]
        for i in range(len(idx_labels)):
            y = idx_labels[i]
            class_data[y].append(i)
        cls_num_list = [len(class_data[i]) for i in range(args.num_classes)]
        clients.append(LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx, dataset_client_clip=clip_data_local_training, cls_num_list=cls_num_list))

    LID_whole = np.zeros(len(y_train))
    loss_whole = np.zeros(len(y_train))
    LID_client = np.zeros(args.num_users)
    loss_accumulative_whole = np.zeros(len(y_train))
    w_locals = []

    uploaded_protos = []
    start = time.perf_counter()
    for idx in range(args.num_users):
        local = clients[idx]
        proto = local.getproto(clip_model)
        uploaded_protos.append(proto)
    global_protos = proto_aggregation(uploaded_protos)

    acc_all = []
    for idx in range(args.num_users):
        local = clients[idx]
        acc = test_proto(local.CLIP_train, clip_model, global_protos, args)
        acc_all.append(acc)

    noisy_set = [i for i, acc in enumerate(acc_all) if acc < 0.5]
    clean_set = [i for i, acc in enumerate(acc_all) if acc > 0.5]
    uploaded_protos = []
    for idx in clean_set:
        local = clients[idx]
        proto = local.protos
        uploaded_protos.append(proto)
    global_protos = proto_aggregation(uploaded_protos)
    dataloadertemp = get_dataloadertemp(uploaded_protos)
    best_model = global_protos_gen_auto(global_protos, dataloadertemp)

    client_cache_keys = []
    client_cache_values = []
    estimated_noisy_level = np.zeros(args.num_users)

    for idx in range(args.num_users):
        local = clients[idx]
        loss_whole[clients[idx].sample_idx], temp_proto, local_cache_keys, local_cache_values, estimated_noisy_level[idx] = get_loss_auto_prompt(local.CLIP_train, clip_model, best_model, len(clients[idx].sample_idx), global_protos)
        uploaded_protos.append(temp_proto)
        client_cache_keys.append(local_cache_keys)
        client_cache_values.append(local_cache_values)

    loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)
    dataloadertemp = get_dataloadertemp(uploaded_protos)
    global_protos = proto_aggregation(uploaded_protos)
    best_model = global_protos_gen_auto(global_protos, dataloadertemp)

    for now, idx in enumerate(noisy_set):
        local = clients[idx]
        local_temp_auto_model = train_auto_model(copy.deepcopy(best_model), client_cache_keys[now],
                                                 client_cache_values[now], local.cls_num_list)
        sample_idx = np.array(list(dict_users[idx]))
        all_outputs = []
        with torch.no_grad():
            for ind, (images2, labels) in enumerate(local.CLIP_train):
                clip_images = images2.to('cuda')
                labels = labels.to('cuda')
                labels = labels.long()
                with torch.no_grad():
                    image_features = clip_model.encode_image(clip_images)
                image_features = image_features.float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                CoOp_logits, all_prompts = local_temp_auto_model(image_features)
                outputs = CoOp_logits.topk(1, 1, True, True)[1].t()
                all_outputs.append(outputs.cpu().numpy())
        all_outputs_array = np.concatenate(all_outputs, axis=1)
        y_train_noisy_temp = np.array(dataset_train.targets)
        y_train_noisy_temp[sample_idx] = all_outputs_array

        loss = np.array(loss_accumulative_whole[sample_idx])
        relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx])]
        y_train_noisy_new = np.array(dataset_train.targets)

        complement_idx = np.setdiff1d(np.arange(len(sample_idx)), relabel_idx)
        y_train_noisy_new[sample_idx[relabel_idx]] = y_train_noisy_temp[sample_idx[relabel_idx]]
        dataset_train.targets = y_train_noisy_new


    netglob = copy.deepcopy(netglob)
    for rnd in range(args.rounds1):

        w_locals, loss_locals = [], []
        for idx in clean_set:
            local = LocalUpdate2(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), epoch=args.local_ep)

            netglob.load_state_dict(copy.deepcopy(w_local))

    criterion = nn.CrossEntropyLoss(reduction='none')


    m = max(int(args.frac * args.num_users), 1)
    prob = [1/args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):

        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:
            local = LocalUpdate2(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), epoch=args.local_ep)
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        print("FedAvg round",rnd,"："," acc=", acc_s2*100)
    torch.cuda.empty_cache()