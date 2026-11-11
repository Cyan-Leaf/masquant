# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
from quantize.nvfp4 import NVFP4Quantizer
import os
import math

FAST_FORWARD = os.getenv("MAS_FAST", "1") == "1"
WQ_CACHE = os.getenv("MAS_WQ_CACHE", "1") == "1"
SHARE_VIDEO_VISION = os.getenv("MAS_VIDEO_SHARE_VISION", "1") == "1"
DISABLE_CMC = os.getenv("MAS_DISABLE_CMC", "0") == "1"


def build_quantizer(params, shape=None):
    params = dict(params)
    fmt = params.pop("format", "int")
    if fmt == "nvfp4":
        return NVFP4Quantizer(block_size=params.get("block_size", 16))
    params.pop("block_size", None)
    if shape is not None:
        return UniformAffineQuantizer(**params, shape=shape)
    return UniformAffineQuantizer(**params)


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
            self,
            org_module: nn.Linear,
            weight_quant_params: dict = {},
            act_quant_params: dict = {},
            disable_input_quant=False,
            support_training=False,
            name="none",
            layer_index=0,
            mode="train"
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.name = name
        self.layer_index = layer_index
        self.register_buffer('weight', org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = build_quantizer(weight_quant_params, shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = build_quantizer(act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        if support_training == True:
            self.all_in_one_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.text_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.vision_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.audio_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
            self.video_smooth_scale = nn.Parameter(torch.ones(self.in_features, dtype=torch.bfloat16))
        else:
            self.register_buffer('all_in_one_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('text_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('vision_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('audio_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))
            self.register_buffer('video_smooth_scale', torch.ones(self.in_features, dtype=torch.bfloat16))

        self.register_buffer("Lv", None)
        self.register_buffer("Rv", None)
        self.register_buffer("La", None)
        self.register_buffer("Ra", None)

        # merged_scales  : 统一scale;   split_scales : 分模态的scale
        self.inference_mode = os.getenv('inference_mode', 'merged_scales')
        # if self.inference_mode == 'split_scales':
        #     print(f'分模态的 scale')
        # else:
        #     print(f'统一 scale')
        #区分是infer模式还是train模式
        self.mode = mode

    def get_masked_value(self, mask_value, input_tensor):
        # 检查形状
        if mask_value is None or mask_value.shape != input_tensor.shape:
            # 使用 torch.zeros_like 创建一个与 input_tensor 相同形状的零张量
            print(f'do not support !!!!')
            import pdb;
            pdb.set_trace()

        # 使用 torch.where 进行掩码选择
        return torch.where(mask_value, input_tensor, torch.zeros_like(input_tensor))

    def find_different_indices(self, tensor1, tensor2):
        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            return None

        # 找到不相等的位置
        not_equal = ~torch.eq(tensor1, tensor2)  # 或者 tensor1 != tensor2
        indices = torch.where(not_equal)

        # 返回索引和对应的值
        return indices, tensor1[indices], tensor2[indices]

    def forward_mas_train(self, input: torch.Tensor, multi_modal_mask=None):
        cur_dtype = input.dtype
        target_dtype = torch.bfloat16
        inference_mode = self.inference_mode
        scale_dtype = self.vision_smooth_scale.dtype
        weight_dtype = self.weight.dtype

        if cur_dtype != target_dtype or cur_dtype != scale_dtype or cur_dtype != weight_dtype:
            print(f'data type is invalid')
        # if self.layer_index == 0:
        #     inference_mode = 'split_scales'

        # 只有是在 prefill 阶段，才有 mask 信息，所以在 decode 阶段，就按照是文本输出的方式来进行计算。
        video_mask = None
        if multi_modal_mask is not None and input.shape[1] != 1:
            if len(multi_modal_mask) == 4:
                (audio_mask, image_mask, text_mask, video_mask) = multi_modal_mask
            else:
                (audio_mask, image_mask, text_mask) = multi_modal_mask
        else:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)
            image_mask = audio_mask
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        if audio_mask is None:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if image_mask is None:
            image_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if text_mask is None:
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        if video_mask is None:
            video_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        # 为了避免出现scale中有含0的情况，需要做一些处理
        # 1e-5 在 bfloat16 中可以很好地表示
        epsilon_value = 1e-5

        # 将 epsilon 转换为与张量相同的类型和设备
        epsilon = torch.tensor(
            epsilon_value,
            dtype=self.vision_smooth_scale.dtype,
            device=self.vision_smooth_scale.device
        )
        # with torch.no_grad():
        #     if inference_mode == 'merged_scales':
        #         self.all_in_one_smooth_scale.data.add_(epsilon)
        #     else:
        #         self.vision_smooth_scale.data.add_(epsilon)
        #         self.audio_smooth_scale.data.add_(epsilon)
        #         self.text_smooth_scale.data.add_(epsilon)

        # 统一scale
        if inference_mode == 'merged_scales':
            if self.use_weight_quant:
                weight_scaled = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.all_in_one_smooth_scale).to(cur_dtype))
            else:
                weight_scaled = self.weight
            bias = self.bias

            if self.use_act_quant and not self.disable_input_quant:
                input_scaled = self.act_quantizer((input.to(target_dtype) / self.all_in_one_smooth_scale).to(cur_dtype))
            else:
                input_scaled = input

            out = self.fwd_func(input_scaled, weight_scaled,
                                bias=self.bias.to(cur_dtype) if self.bias is not None else None)
            out = out.to(cur_dtype)
            # if not math.isfinite(self.all_in_one_smooth_scale[0].item()):
            #     import pdb; pdb.set_trace()

        # 分模态scale
        else:
            # self.audio_smooth_scale = self.text_smooth_scale
            # self.vision_smooth_scale = self.text_smooth_scale
            if self.use_weight_quant:
                weight_audio = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.audio_smooth_scale).to(cur_dtype))
                weight_vision = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.vision_smooth_scale).to(cur_dtype))
                weight_text = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.text_smooth_scale).to(cur_dtype))
                weight_video = self.weight_quantizer(
                    (self.weight.to(target_dtype) * self.video_smooth_scale).to(cur_dtype))
            else:
                weight_audio = (self.weight.to(target_dtype) * self.audio_smooth_scale).to(cur_dtype)
                weight_vision = (self.weight.to(target_dtype) * self.vision_smooth_scale).to(cur_dtype)
                weight_text = (self.weight.to(target_dtype) * self.text_smooth_scale).to(cur_dtype)
                weight_video = (self.weight.to(target_dtype) * self.video_smooth_scale).to(cur_dtype)

            if self.use_act_quant and not self.disable_input_quant:
                input_audio = self.act_quantizer(
                    self.get_masked_value(audio_mask, input.to(target_dtype)) / (self.audio_smooth_scale + epsilon)).to(
                    cur_dtype)
                input_vision = self.act_quantizer(self.get_masked_value(image_mask, input.to(target_dtype)) / (
                            self.vision_smooth_scale + epsilon)).to(cur_dtype)
                input_text = self.act_quantizer(
                    self.get_masked_value(text_mask, input.to(target_dtype)) / (self.text_smooth_scale + epsilon)).to(
                    cur_dtype)
                input_video = self.act_quantizer(
                    self.get_masked_value(video_mask, input.to(target_dtype)) / (self.video_smooth_scale + epsilon)).to(
                    cur_dtype)
            else:
                input_audio = (self.get_masked_value(audio_mask, input.to(target_dtype)) / (
                            self.audio_smooth_scale + epsilon)).to(cur_dtype)
                input_vision = (self.get_masked_value(image_mask, input.to(target_dtype)) / (
                            self.vision_smooth_scale + epsilon)).to(cur_dtype)
                input_text = (self.get_masked_value(text_mask, input.to(target_dtype)) / (
                            self.text_smooth_scale + epsilon)).to(cur_dtype)
                input_video = (self.get_masked_value(video_mask, input.to(target_dtype)) / (
                            self.video_smooth_scale + epsilon)).to(cur_dtype)

            out_audio = self.fwd_func(input_audio, weight_audio, bias=None)
            out_vision = self.fwd_func(input_vision, weight_vision, bias=None)
            out_text = self.fwd_func(input_text, weight_text,
                                     bias=self.bias.to(cur_dtype) if self.bias is not None else None)
            out_video = self.fwd_func(input_video, weight_video, bias=None)
            # if input_vision.mean().item() <= 0.0000001:
            # import pdb;pdb.set_trace()
            # print(f'layer_{self.layer_index}, name: {self.name}, text_smooth_scale: {self.text_smooth_scale.mean()}, audio_smooth_scale: {self.audio_smooth_scale.mean()}, vision_smooth_scale: {self.vision_smooth_scale.mean()}')
            # print(f'layer_{self.layer_index}, name: {self.name}, out_text: {out_text.mean()}, out_audio: {out_audio.mean()}, out_vision: {out_vision.mean()}')
            out = out_text + out_vision + out_audio + out_video
            if torch.isnan(out_vision).any() or torch.isnan(out_text).any() or torch.isnan(out_audio).any():
                import pdb;
                pdb.set_trace()

            out = out.to(cur_dtype)

        return out

        # return out, input_text, input_vision, input_audio, weight_text, weight_vision, weight_audio

    def forward_mas_infer(self, input: torch.Tensor, multi_modal_mask=None):
        # 三模态补偿模式
        cur_dtype = input.dtype
        target_dtype = torch.bfloat16

        # 只有是在 prefill 阶段，才有 mask 信息，所以在 decode 阶段，就按照是文本输出的方式来进行计算。
        if multi_modal_mask is not None and input.shape[1] != 1:
            (audio_mask, image_mask, text_mask) = multi_modal_mask[:3]
            if len(multi_modal_mask) == 4 and multi_modal_mask[3] is not None:
                video_mask = multi_modal_mask[3]
                image_mask = video_mask if image_mask is None else (image_mask | video_mask)
        else:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)
            image_mask = audio_mask
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        if audio_mask is None:
            audio_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if image_mask is None:
            image_mask = torch.full(input.shape, False, dtype=torch.bool, device=input.device)

        if text_mask is None:
            text_mask = torch.full(input.shape, True, dtype=torch.bool, device=input.device)

        # 分模态scale, 权重已经提前量化好
        q_input_audio = self.act_quantizer(
            self.get_masked_value(audio_mask, input.to(target_dtype)) / self.audio_smooth_scale).to(cur_dtype)
        q_input_vision = self.act_quantizer(
            self.get_masked_value(image_mask, input.to(target_dtype)) / self.vision_smooth_scale).to(cur_dtype)
        q_input_text = self.act_quantizer(
            self.get_masked_value(text_mask, input.to(target_dtype)) / self.text_smooth_scale).to(cur_dtype)

        input_audio = (self.get_masked_value(audio_mask, input.to(target_dtype)) / self.audio_smooth_scale).to(
            cur_dtype)
        input_vision = (self.get_masked_value(image_mask, input.to(target_dtype)) / self.vision_smooth_scale).to(
            cur_dtype)

        out_audio = self.fwd_func(q_input_audio, self.q_weight, bias=None)
        out_vision = self.fwd_func(q_input_vision, self.q_weight, bias=None)
        out_text = self.fwd_func(q_input_text, self.q_weight,
                                 bias=self.bias.to(cur_dtype) if self.bias is not None else None)  # 这里其实可以简化，但考虑到数值一致性。
        out = out_text + out_vision + out_audio

        if self.Lv is not None:
            # vision低秩分解的另一条旁路
            xvL = torch.matmul(input_vision, self.Lv)
            xvLR = torch.matmul(xvL, self.Rv).to(cur_dtype)
        else:
            xvLR = 0

        if self.La is not None:
            # audio低秩分解的另一条旁路
            xaL = torch.matmul(input_audio, self.La)
            xaLR = torch.matmul(xaL, self.Ra).to(cur_dtype)
        else:
            xaLR = 0

        out = out + xvLR + xaLR
        out = out.to(cur_dtype)

        # weight_vision = self.q_weight.double() + torch.matmul(self.Lv, self.Rv).T
        # weight_audio = self.q_weight.double() + torch.matmul(self.La, self.Ra).T
        # return out, input_text, input_vision, input_audio, self.q_weight, weight_vision.to(cur_dtype), weight_audio.to(cur_dtype)
        return out

    def _token_groups(self, input, multi_modal_mask):
        """把 (audio, image, text[, video]) 掩码转成 {模态: 行索引}，行索引基于 input.reshape(-1, D)。"""
        n = input.shape[:-1].numel()
        dev = input.device
        if multi_modal_mask is None or input.shape[1] == 1:
            return {"text": None}, n
        mm = list(multi_modal_mask)
        audio_m, image_m, text_m = mm[0], mm[1], mm[2]
        video_m = mm[3] if len(mm) == 4 else None

        def tok(m):
            if m is None:
                return None
            if m.dim() == input.dim():
                m = m[..., 0]
            return m.reshape(-1).to(dev)

        audio_t, image_t, text_t, video_t = tok(audio_m), tok(image_m), tok(text_m), tok(video_m)
        if text_t is None:
            text_t = torch.ones(n, dtype=torch.bool, device=dev)
            for t in (audio_t, image_t, video_t):
                if t is not None:
                    text_t = text_t & ~t
        if video_t is not None and SHARE_VIDEO_VISION:
            image_t = video_t if image_t is None else (image_t | video_t)
            video_t = None
        groups = {}
        for name, t in (("text", text_t), ("vision", image_t), ("audio", audio_t), ("video", video_t)):
            if t is None:
                continue
            idx = t.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                groups[name] = idx
        if len(groups) == 1 and "text" in groups and groups["text"].numel() == n:
            groups = {"text": None}
        return groups, n

    def _scale_of(self, name):
        # CMC 推理（mode=infer）本质就是分模态平滑，与 inference_mode 环境变量无关；
        # 否则 infer_mas.py 未设环境变量时会取到从未赋值的 all_in_one_smooth_scale（全 1）
        if self.mode != "infer" and self.inference_mode == 'merged_scales':
            return self.all_in_one_smooth_scale
        return {"text": self.text_smooth_scale, "vision": self.vision_smooth_scale,
                "audio": self.audio_smooth_scale, "video": self.video_smooth_scale}[name]

    def clear_weight_cache(self):
        self.__dict__.pop("_wq_cache", None)

    def _apply(self, fn, *args, **kwargs):
        # .to()/.cpu()/.cuda() 都走 _apply；缓存不是 buffer，不会跟着搬，必须丢掉以免泄漏在原设备上
        self.clear_weight_cache()
        return super()._apply(fn, *args, **kwargs)

    def _smoothed_weight(self, scale, tag, cur_dtype):
        target_dtype = torch.bfloat16

        def build():
            w = (self.weight.to(target_dtype) * scale).to(cur_dtype)
            # LWC 的裁剪因子是 fp32，量化结果会被提升为 fp32；统一转回激活精度
            return self.weight_quantizer(w).to(cur_dtype) if self.use_weight_quant else w

        if not WQ_CACHE or torch.is_grad_enabled():
            return build()
        lwc_ver = tuple(p._version for n, p in self.weight_quantizer.named_parameters())
        key = (tag, scale.data_ptr(), scale._version, self.weight.data_ptr(), self.weight._version,
               self.use_weight_quant, lwc_ver, cur_dtype)
        # 记录当前是否有需要梯度的权重缩放参数；一旦从 no-grad 切到 grad，缓存必须作废
        key = key + (self.mode, torch.is_grad_enabled())
        cache = self.__dict__.setdefault("_wq_cache", {})
        hit = cache.get(tag)
        if hit is not None and hit[0] == key:
            return hit[1]
        w = build()
        cache[tag] = (key, w)
        return w

    def _quant_act(self, x, scale, cur_dtype, eps):
        target_dtype = torch.bfloat16
        xs = x.to(target_dtype) / (scale + eps)
        if self.use_act_quant and not self.disable_input_quant:
            xs = self.act_quantizer(xs)
        return xs.to(cur_dtype)

    def forward_mas_train_fast(self, input: torch.Tensor, multi_modal_mask=None):
        """与 forward_mas_train 数值等价：被 mask 掉的行在原实现里是零输入→零输出，这里直接只算本模态的行。"""
        cur_dtype = input.dtype
        eps = torch.tensor(1e-5, dtype=self.vision_smooth_scale.dtype, device=self.vision_smooth_scale.device)
        bias = self.bias.to(cur_dtype) if self.bias is not None else None
        if self.inference_mode == 'merged_scales':
            s = self.all_in_one_smooth_scale
            w = self._smoothed_weight(s, "merged", cur_dtype) if self.use_weight_quant else self.weight
            if self.use_act_quant and not self.disable_input_quant:
                x = self.act_quantizer((input.to(torch.bfloat16) / s).to(cur_dtype))
            else:
                x = input
            return self.fwd_func(x, w, bias=bias).to(cur_dtype)
        groups, n = self._token_groups(input, multi_modal_mask)

        if "text" in groups and groups["text"] is None:
            s = self._scale_of("text")
            w = self._smoothed_weight(s, "text", cur_dtype)
            x = self._quant_act(input, s, cur_dtype, eps)
            return self.fwd_func(x, w, bias=bias).to(cur_dtype)

        x2 = input.reshape(-1, input.shape[-1])
        out = None
        for name, idx in groups.items():
            s = self._scale_of(name)
            w = self._smoothed_weight(s, name, cur_dtype)
            y = self.fwd_func(self._quant_act(x2.index_select(0, idx), s, cur_dtype, eps), w, bias=bias)
            if out is None:
                out = torch.zeros(x2.shape[0], y.shape[-1], dtype=y.dtype, device=y.device)
            out = out.index_copy(0, idx, y)
        return out.reshape(*input.shape[:-1], -1).to(cur_dtype)

    def forward_mas_infer_fast(self, input: torch.Tensor, multi_modal_mask=None):
        """CMC 推理：各模态用自己的激活平滑因子、共享一份 Q(S_t W)，非文本模态加低秩补偿。
        由于激活是 per-token 量化，可以把各行除以各自的 s_m 后拼成一个矩阵，只做一次主 GEMM。"""
        cur_dtype = input.dtype
        target_dtype = torch.bfloat16
        bias = self.bias.to(cur_dtype) if self.bias is not None else None
        if os.getenv("MAS_DISABLE_CMC", "0") == "1" or os.getenv("MAS_NO_CMC", "0") == "1":
            xq = self.act_quantizer(input.reshape(-1, input.shape[-1]).to(target_dtype) / self.text_smooth_scale)
            out = self.fwd_func(xq.to(cur_dtype), self.q_weight, bias=bias)
            return out.reshape(*input.shape[:-1], -1).to(cur_dtype)
        groups, n = self._token_groups(input, multi_modal_mask)
        x2 = input.reshape(-1, input.shape[-1]).to(target_dtype)

        if "text" in groups and groups["text"] is None:
            xs = x2 / self.text_smooth_scale
            smoothed = {}
        else:
            xs = torch.empty_like(x2)
            smoothed = {}
            for name, idx in groups.items():
                part = x2.index_select(0, idx) / self._scale_of(name)
                xs.index_copy_(0, idx, part)
                smoothed[name] = (idx, part)
        xq = self.act_quantizer(xs).to(cur_dtype)
        out = self.fwd_func(xq, self.q_weight, bias=bias)

        for name, L, R in (("vision", self.Lv, self.Rv), ("audio", self.La, self.Ra)):
            if L is None or name not in smoothed:
                continue
            idx, part = smoothed[name]
            corr = torch.matmul(torch.matmul(part.to(L.dtype), L), R).to(out.dtype)
            out = out.index_add(0, idx, corr)
        return out.reshape(*input.shape[:-1], -1).to(cur_dtype)

    def forward(self, input: torch.Tensor, multi_modal_mask=None):
        if self.mode == "infer":
            if FAST_FORWARD:
                return self.forward_mas_infer_fast(input, multi_modal_mask)
            return self.forward_mas_infer(input, multi_modal_mask)
        else:
            if FAST_FORWARD:
                return self.forward_mas_train_fast(input, multi_modal_mask)
            return self.forward_mas_train(input, multi_modal_mask)
        # # a1, b1, c1, d1, e1, f1, g1 = self.forward_mas_train(input, multi_modal_mask)
        # # a2, b2, c2, d2, e2, f2, g2 = self.forward_mas_infer(input, multi_modal_mask)
        #
        # # return self.forward_mas_train(input, multi_modal_mask)
        # return self.forward_mas_infer(input, multi_modal_mask)
        # # return a2

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
