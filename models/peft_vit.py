import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip.model import VisionTransformer as CLIP_ViT
from timm.models.vision_transformer import VisionTransformer as ViT

from .peft_modules import *


class ViT_Tuner(nn.Module):
    """ All instance variables in this class will be optimized.
    """
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()
        dtype = vit_model.patch_embed.proj.weight.dtype

        use_full_tuning = cfg.full_tuning
        use_bias_tuning = cfg.bias_tuning
        use_ln_tuning = cfg.ln_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_adapter = cfg.adapter
        use_adaptformer = cfg.adaptformer
        use_lora = cfg.lora
        use_ssf_attn = cfg.ssf_attn
        use_ssf_mlp = cfg.ssf_mlp
        use_ssf_ln = cfg.ssf_ln
        partial = cfg.partial
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim


        if (use_vpt_shallow or use_vpt_deep) and (vpt_len is None):
            vpt_len = 10
            print("Visual prompt length set to {}".format(vpt_len))

        if (use_adapter or use_adaptformer or use_lora) and (adapter_dim is None):

            adapter_dim = 4
            print("Adapter bottle dimension set to {}".format(adapter_dim))


        block_tuned = None
        bias_tuned = None
        ln_tuned = None
        vpt_list = nn.ModuleList([None])
        adapter_list = nn.ModuleList([None])

        if use_adaptformer:
            adaptformer_list1 = nn.ModuleList([
                *[AdaptFormer(in_dim=192, bottle_dim=adapter_dim, dtype=dtype) for _ in range(2)]
            ])
            adaptformer_list2 = nn.ModuleList([
                *[AdaptFormer(in_dim=384, bottle_dim=adapter_dim, dtype=dtype) for _ in range(2)]
            ])
            adaptformer_list3 = nn.ModuleList([
                *[AdaptFormer(in_dim=768, bottle_dim=adapter_dim, dtype=dtype) for _ in range(18)]
            ])
            adaptformer_list4 = nn.ModuleList([
                *[AdaptFormer(in_dim=1536, bottle_dim=adapter_dim, dtype=dtype) for _ in range(2)]
            ])
            adaptformer_list = nn.ModuleList([
                adaptformer_list1,adaptformer_list2,adaptformer_list3,adaptformer_list4
            ])


        lora_list = nn.ModuleList([None])
        ssf_attn_list = nn.ModuleList([None])
        ssf_mlp_list = nn.ModuleList([None])
        ssf_ln_list = nn.ModuleList([None])
        
        # To be optimized
        self.block_tuned = block_tuned
        self.bias_tuned = bias_tuned
        self.ln_tuned = ln_tuned
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.adaptformer_list = adaptformer_list
        self.lora_list = lora_list
        self.ssf_attn_list = ssf_attn_list
        self.ssf_mlp_list = ssf_mlp_list
        self.ssf_ln_list = ssf_ln_list


class ViT_Tuner_peft_dim(nn.Module):
    """ All instance variables in this class will be optimized.
    """

    def __init__(self, cfg, vit_model, num_classes,peft_dim):
        super().__init__()
        #dtype = vit_model.patch_embed.proj.weight.dtype
        dtype = torch.float32

        use_full_tuning = cfg.full_tuning
        use_bias_tuning = cfg.bias_tuning
        use_ln_tuning = cfg.ln_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_adapter = cfg.adapter
        use_adaptformer = cfg.adaptformer
        use_lora = cfg.lora
        use_ssf_attn = cfg.ssf_attn
        use_ssf_mlp = cfg.ssf_mlp
        use_ssf_ln = cfg.ssf_ln
        partial = cfg.partial
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim


        if (use_vpt_shallow or use_vpt_deep) and (vpt_len is None):
            vpt_len = 10
            print("Visual prompt length set to {}".format(vpt_len))

        if (use_adapter or use_adaptformer or use_lora) and (adapter_dim is None):
            adapter_dim = 4
            print("Adapter bottle dimension set to {}".format(adapter_dim))


        adapter_dim = peft_dim

        block_tuned = None
        bias_tuned = None
        ln_tuned = None
        vpt_list = nn.ModuleList([None])
        adapter_list = nn.ModuleList([None])

        if use_adaptformer:
            adaptformer_list1 = nn.ModuleList([
                *[AdaptFormer(in_dim=192, bottle_dim=adapter_dim, dtype=dtype) for _ in range(2)]
            ])
            adaptformer_list2 = nn.ModuleList([
                *[AdaptFormer(in_dim=384, bottle_dim=adapter_dim, dtype=dtype) for _ in range(2)]
            ])
            adaptformer_list3 = nn.ModuleList([
                *[AdaptFormer(in_dim=768, bottle_dim=adapter_dim, dtype=dtype) for _ in range(18)]
            ])
            adaptformer_list4 = nn.ModuleList([
                *[AdaptFormer(in_dim=1536, bottle_dim=adapter_dim, dtype=dtype) for _ in range(2)]
            ])
            adaptformer_list = nn.ModuleList([
                adaptformer_list1, adaptformer_list2, adaptformer_list3, adaptformer_list4
            ])

        lora_list = nn.ModuleList([None])
        ssf_attn_list = nn.ModuleList([None])
        ssf_mlp_list = nn.ModuleList([None])
        ssf_ln_list = nn.ModuleList([None])

        # To be optimized
        self.block_tuned = block_tuned
        self.bias_tuned = bias_tuned
        self.ln_tuned = ln_tuned
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.adaptformer_list = adaptformer_list
        self.lora_list = lora_list
        self.ssf_attn_list = ssf_attn_list
        self.ssf_mlp_list = ssf_mlp_list
        self.ssf_ln_list = ssf_ln_list


class Peft_ViT(nn.Module):
    def __init__(self, vit_model):
        super().__init__()

        if isinstance(vit_model, CLIP_ViT):
            self.backbone = "CLIP-VIT"
            self.patch_embedding = vit_model.conv1
            self.class_embedding = vit_model.class_embedding
            self.positional_embedding = vit_model.positional_embedding
            self.ln_pre = vit_model.ln_pre
            self.blocks = vit_model.transformer.resblocks
            self.ln_post = vit_model.ln_post
            self.proj = vit_model.proj  # not used
            self.out_dim = self.ln_post.bias.shape[0]
            # self.out_dim = self.proj.shape[1]
        
        elif isinstance(vit_model, ViT):
            self.backbone = "ViT"
            self.patch_embedding = vit_model.patch_embed.proj
            self.class_embedding = vit_model.cls_token
            self.positional_embedding = vit_model.pos_embed
            self.ln_pre = vit_model.norm_pre
            self.blocks = vit_model.blocks
            self.ln_post = vit_model.norm
            self.proj = nn.Identity()
            self.out_dim = self.ln_post.bias.shape[0]

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype



    def forward(self, x, tuner=None, head=None):
        x = x.to(self.dtype)
        x = self.patch_embedding(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                adaptformer = tuner.adaptformer_list[i]
                lora = tuner.lora_list[i]
                ssf_attn = tuner.ssf_attn_list[i]
                ssf_mlp = tuner.ssf_mlp_list[i]
                ssf_ln = tuner.ssf_ln_list[i]
            else:
                vpt = adapter = adaptformer = lora = ssf_attn = ssf_mlp = ssf_ln = None

            if vpt is not None:
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            if self.backbone == "CLIP-VIT":
                _attn = block.attn
                _ln_1 = block.ln_1
                _mlp = block.mlp
                _ln_2 = block.ln_2

                _attn_in_proj_weight = _attn.in_proj_weight
                _attn_in_proj_bias = _attn.in_proj_bias
                _attn_out_proj_weight = _attn.out_proj.weight
                _attn_out_proj_bias = _attn.out_proj.bias
                _mlp_in_proj = _mlp[0]
                _mlp_act = _mlp[1]
                _mlp_out_proj = _mlp[2]

                _num_heads = _attn.num_heads
                _head_dim = _emb_dim // _num_heads
            
            elif self.backbone == "ViT":
                _attn = block.attn
                _ln_1 = block.norm1
                _mlp = block.mlp
                _ln_2 = block.norm2

                _attn_in_proj_weight = _attn.qkv.weight
                _attn_in_proj_bias = _attn.qkv.bias
                _attn_out_proj_weight = _attn.proj.weight
                _attn_out_proj_bias = _attn.proj.bias
                _mlp_in_proj = _mlp.fc1
                _mlp_act = _mlp.act
                _mlp_out_proj = _mlp.fc2

                _num_heads = _attn.num_heads
                _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x  # deep copy

            x = _ln_1(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_1"](x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            if ssf_attn is not None:
                qkv = ssf_attn["attn_in"](qkv)

            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            x = F.scaled_dot_product_attention(q, k, v)
            # scaled_dot_product_attention:
            # q = q / math.sqrt(_head_dim)
            # attn = torch.bmm(q, k.transpose(-2, -1))
            # attn = F.softmax(attn, dim=-1)
            # x = torch.bmm(attn, v)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if ssf_attn is not None:
                x = ssf_attn["attn_out"](x)

            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = x + identity

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x  # deep copy

            x = _ln_2(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_2"](x)

            x = _mlp_in_proj(x)
            if ssf_mlp is not None:
                x = ssf_mlp["mlp_in"](x)
            
            x = _mlp_act(x)

            x = _mlp_out_proj(x)
            if ssf_mlp is not None:
                x = ssf_mlp["mlp_out"](x)
            
            if adapter is not None:
                x = x + adapter(x)
            
            if adaptformer is not None:
                x = x + adaptformer(identity)
            
            x = x + identity
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]
        x = self.ln_post(x)
        # x = x @ self.proj

        if head is None:
            return x
        else:
            return head(x)
