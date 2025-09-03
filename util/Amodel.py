import clip
import numpy as np
import torch
from torch import nn


import torch.nn.functional as F


class AutoPromptModel(nn.Module):
    def __init__(self, visual_proto_emb, channel_num=512):
        super(AutoPromptModel, self).__init__()

        self.channel_num = channel_num
        self.prototypes = visual_proto_emb

        self.query = nn.Linear(self.channel_num, self.channel_num)
        self.key = nn.Linear(self.channel_num, self.channel_num)
        self.value = nn.Linear(self.channel_num, self.channel_num)
        self.proj = nn.Linear(self.channel_num, self.channel_num)
        self.out = nn.Linear(self.channel_num, self.channel_num)

        nn.init.normal_(self.query.weight, mean=0, std=np.sqrt(2.0 / (self.channel_num)))
        nn.init.normal_(self.key.weight, mean=0, std=np.sqrt(2.0 / (self.channel_num)))
        nn.init.normal_(self.value.weight, mean=0, std=np.sqrt(2.0 / (self.channel_num)))

        nn.init.xavier_normal_(self.out.weight)
        self.norm1 = nn.LayerNorm(self.channel_num)
        self.dropout = nn.Dropout(0.1)


    def infer(self, x):

        q, k, v = self.query(x), self.key(self.prototypes), self.value(self.prototypes)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.channel_num) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(attn, v)
        att_weights = self.out(v) + self.proj(x)
        visual_att_weights = self.norm1(att_weights)

        weights = visual_att_weights
        residual_weights = weights.unsqueeze(1)
        all_prompts = residual_weights
        all_prompts = all_prompts / all_prompts.norm(dim=-1, keepdim=True)

        return all_prompts

    def forward(self, x):

        q, k, v = self.query(x), self.key(self.prototypes), self.value(self.prototypes)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.channel_num) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(attn, v)
        att_weights = self.out(v) + self.proj(x)
        visual_att_weights = self.norm1(att_weights)

        weights = visual_att_weights
        residual_weights = weights.unsqueeze(1)
        text_prompt = self.prototypes.unsqueeze(0)

        all_prompts = residual_weights + text_prompt
        all_prompts = all_prompts / all_prompts.norm(dim=-1, keepdim=True)

        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)

        logits = all_prompts @ x
        logits = logits.squeeze(-1)

        return 100. * logits, all_prompts

