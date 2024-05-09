# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import math
from einops import rearrange

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from pruning.merge import bipartite_soft_matching, merge_source, merge_wavg
from pruning.utils import parse_r
import sys
from pruning.merge import prune_source


class PruningBlock(Block):
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.size()

        keep_head_idx, rm_head_idx = None, None
        x_attn, weights = self.attn(self.norm1(x), keep_head_idx, rm_head_idx)
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x)))

        scores = weights[...,0,:]

        if self._prune_info['method']  == "tk":
            ##tk model
            scores[:,0] = math.inf
            #tokenのベクトルの大きさを降順で並べる
            idx = scores.argsort(descending = True)
            #下位 n - k個以外の token を指定する
            keep_num = n - self.k if n - self.k >= self.protect_tokens else self.protect_tokens
            keep_idx = idx[..., :keep_num]
            x = x.gather(dim=-2, index = keep_idx.unsqueeze(-1).repeat(1,1,d))

        elif self._prune_info['method'] == "evit":
            #evit model
            scores[:,0] = math.inf
            #tokenのベクトルの大きさを降順で並べる
            idx = scores.argsort(descending = True)
            #下位 n - k個以外の token を指定する
            keep_num = n - self.k if n - self.k >= self.protect_tokens else self.protect_tokens
            keep_idx = idx[..., :keep_num]
            keep_x = x.gather(dim=-2, index = keep_idx.unsqueeze(-1).repeat(1,1,d))

            #下位 n-k個の token を指定する
            del_y = idx[...,keep_num:]
            #指定した下位n-k個の token 特徴インデックスの要素を移す
            del_scores = scores.gather(dim=-1, index = del_y)
            del_x = x.gather(dim=-2, index = del_y.unsqueeze(-1).repeat(1,1,d))

            #要素積の実装をすることで、各 token の特徴量に重み付けをする
            x_ = del_scores.unsqueeze(-1)*del_x

            #重み付けしたそれぞれのtokenを一つの token にする
            sum_x = torch.sum(x_ ,dim=1).unsqueeze(-2)
            x = torch.cat((keep_x,sum_x), dim = -2)

        elif self._prune_info['method'] == "tk_n":
            # topk norm model
            with torch.no_grad():
                #tokenをベクトルの大きさで見る
                scores = torch.norm(x, dim= -1)
            #class tokenを判別できるように最大にする
            scores[:,0] = math.inf
            #tokenのベクトルの大きさを降順で並べる
            idx = scores.argsort(descending = True)
            #下位 n - k個以外の token を指定する
            keep_num = n - self.k if n - self.k >= self.protect_tokens else self.protect_tokens
            keep_idx = idx[..., :keep_num]
            x = x.gather(dim=-2, index = keep_idx.unsqueeze(-1).repeat(1,1,d))

        elif self._prune_info['method'] == "evit_n":
            # EViT norm
            with torch.no_grad():
                #tokenをベクトルの大きさで見る
                scores = torch.norm(x, dim= -1)
            #class tokenを判別できるように最大にする
            scores[:,0] = math.inf
            #tokenのベクトルの大きさを降順で並べる
            idx = scores.argsort(descending = True)
            #下位 n - k個以外の token を指定する
            keep_num = n - self.k if n - self.k >= self.protect_tokens else self.protect_tokens
            keep_idx = idx[...,:keep_num]
            keep_x = x.gather(dim=-2, index = keep_idx.unsqueeze(-1).repeat(1,1,d))

            #下位 n-k個の token を指定する
            del_y = idx[...,keep_num:]
            #指定した下位n-k個の token インデックスの要素を移す
            del_norm = scores.gather(dim=-1, index = del_y)
            del_x = x.gather(dim=-2, index = del_y.unsqueeze(-1).repeat(1,1,d))

            #softmax関数に下位n-k個norm後のtokenを通すことで、それぞれの tokenが削除部分に占める重要性の比率を求める
            del_scores = F.softmax(del_norm, dim=-1)

            #要素積の実装をすることで、各 token の特徴量に重み付けをする
            x_ = del_scores.unsqueeze(-1)*del_x

            #重み付けしたそれぞれのtokenを一つの token にする
            sum_x = torch.sum(x_ ,dim=1).unsqueeze(-2)
            x = torch.cat((keep_x,sum_x), dim = -2)


        elif self._prune_info['method'] == "n_wa":
            #ours norm to attention combine
            scores[:,0] = math.inf
            with torch.no_grad():
                #tokenをベクトルの大きさで見る
                norm_scores = torch.norm(x, dim= -1)
            #class tokenを判別できるように最大にする
            norm_scores[:,0] = math.inf
            #tokenのベクトルの大きさを降順で並べる
            idx = norm_scores.argsort(descending = True)
            #下位 n - k個以外の token を指定する　残す部分
            keep_num = n - self.k if n - self.k >= self.protect_tokens else self.protect_tokens
            keep_idx = idx[...,:keep_num]
            keep_x = x.gather(dim=-2, index = keep_idx.unsqueeze(-1).repeat(1,1,d))

            #下位 n-k個の token を指定する 削除する部分
            del_y = idx[...,keep_num:]
            #指定した下位n-k個のインデックスを残す
            del_x = x.gather(dim=-2, index = del_y.unsqueeze(-1).repeat(1,1,d))
            #指定した下位n-k個の token 特徴インデックスの要素を移す
            del_scores = scores.gather(dim=-1, index = del_y)

            #要素積の実装をすることで、各 token の特徴量に重み付けをする
            x_ = del_scores.unsqueeze(-1)*del_x
            #重み付けしたそれぞれのtokenを一つの token にする
            sum_x = torch.sum(x_ ,dim=1).unsqueeze(-2)
            x = torch.cat((keep_x,sum_x), dim = -2)

        elif self._prune_info['method'] == "a_wn":
            # attention to ours norm combine
            scores[:,0] = math.inf
            with torch.no_grad():
                #tokenをベクトルの大きさで見る
                norm_scores = torch.norm(x, dim= -1)
            #class tokenを判別できるように最大にする
            norm_scores[:,0] = math.inf
            #tokenのベクトルの大きさを降順で並べる
            idx = scores.argsort(descending = True)
            #下位 n - k個以外の token を指定する　残す部分
            keep_num = n - self.k if n - self.k >= self.protect_tokens else self.protect_tokens
            keep_idx = idx[...,:keep_num]
            keep_x = x.gather(dim=-2, index = keep_idx.unsqueeze(-1).repeat(1,1,d))

            #下位 n-k個の token を指定する 削除する部分
            del_y = idx[...,keep_num:]
            #指定した下位n-k個のインデックスを残す
            del_x = x.gather(dim=-2, index = del_y.unsqueeze(-1).repeat(1,1,d))
            #指定した下位n-k個の token 特徴インデックスの要素を移す
            del_norm = norm_scores.gather(dim=-1, index = del_y)

            #softmax関数に下位n-k個のtokenを通すことで、それぞれの tokenが削除部分に占める比率を求める
            del_scores = F.softmax(del_norm, dim=-1)
            #要素積の実装をすることで、各 token の特徴量に重み付けをする
            x_ = del_scores.unsqueeze(-1)*del_x
            #重み付けしたそれぞれのtokenを一つの token にする
            sum_x = torch.sum(x_ ,dim=1).unsqueeze(-2)
            x = torch.cat((keep_x,sum_x), dim = -2)

        self._prune_info['source'] = prune_source(self._prune_info['source'], keep_idx, args = self._prune_info)

        return x


class PruningAttention(Attention):
    def forward(
        self, x: torch.Tensor, select_method: str, keep_head_idx: torch.Tensor = None, rm_head_idx: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # q = q.gather(-3, index=keep_head_idx)
        # k = k.gather(-3, index=keep_head_idx)
        # v_keep = v.gather(-3, index=keep_head_idx)
        # v_rm = v.gather(-3, index=rm_head_idx)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # v = (attn @ v_keep)
        # x = torch.cat([v_keep, v_rm], dim=-3).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn.mean(1)


def make_pruning_class(transformer_class):
    class PrunigVisionTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._prune_info['source'] = None
            self._prune_info['img_shape'] = args[0].shape
            self._prune_info['n_token'] = (args[0].shape[-1] // self._prune_info['patch_size'])**2 + 1
            return super().forward(*args, **kwdargs)

    return PrunigVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
, pool_k = 10, select_method = "tk"):
    PruningVisionTransformer = make_pruning_class(model.__class__)

    model.__class__ = PruningVisionTransformer

    model._prune_info = {
        'source' : None,
        'patch_size' : 16,
        'merge' : True,
        'method': select_method,
    }

    if not pool_k or type(pool_k) is int:
        pool_k = [pool_k for _ in range(len(model.blocks))]
    assert len(pool_k) == len(model.blocks)

    pool_k.reverse()
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = PruningBlock
            module.k = pool_k.pop()
            module.protect_tokens = 2
            module._prune_info = model._prune_info

        elif isinstance(module, Attention):
            module.__class__ = PruningAttention
