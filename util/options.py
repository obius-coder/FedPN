# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds1', type=int, default=1, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=1, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs")
    parser.add_argument('--frac', type=float, default=1, help="fration of selected clients in preprocessing stage")

    parser.add_argument('--num_users', type=int, default=50, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")

    # noise arguments
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")


    parser.add_argument('--noise_type',type=str, default='pt', choices=['symmetric', 'pairflip', 'pt'], help='noise type of each clients')
    parser.add_argument('--noise_rate', type=float, default=0.05,  help="noise rate of each clients")



    parser.add_argument('--model', type=str, default='cnn', help="model name")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--alpha_dirichlet', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    return parser.parse_args()
