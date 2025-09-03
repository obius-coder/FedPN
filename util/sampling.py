# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users


def dirnoniid(y_train, num_classes, num_users, alpha_dirichlet = 0.5):

    min_size = 0
    K = num_classes
    N = len(y_train)
    dict_users = {}
    try_cnt = 1
    least_samples = 1
    while min_size < least_samples:
        if try_cnt > 1:
            print(
                f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha_dirichlet, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        try_cnt += 1

    for j in range(num_users):
        dict_users[j] = idx_batch[j]
    return dict_users