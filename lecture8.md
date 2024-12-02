---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
# backgroundImage: url('https://marp.app/assets/hero-background.svg')
marp: true
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
a[href='red'] {
    color: red;
    pointer-events: none;
    cursor: default;
    text-decoration: none;
}
</style>

<style>
img[alt~="right"] {
  display: block;
  margin:auto;
}
a[href='red'] {
    color: red;
    pointer-events: none;
    cursor: default;
    text-decoration: none;
}
</style>


![bg left:45% 80%](images/course.webp)

# **LLM智能应用开发**

第8讲: 大语言模型解析 V

基于HF LlaMA实现的讲解

<!-- https://marp.app/ -->

---

# LLM结构的学习路径

* LLM结构解析(开源LlaMA)
* 自定义数据集构造
* 自定义损失函数和模型训练/微调

---

# Transformer经典结构

<!-- ![bg right:40% 100%](images/l4/transformer.png) -->

![bg right:30% 100%](images/l4/llama_arch_rope.png)

* Encoder-decoder结构
* 输入部分
  * Input embedding
  * Positional embedding
* Transformer部分
  * Feed forward network
  * Attention module
    * Flash Attention

---

# Transformer经典结构

<!-- ![bg right:40% 100%](images/l4/transformer.png) -->

![bg right:30% 100%](images/l4/llama_arch_rope.png)

* Others
  * Transformers结构改动
    * Mixture-of-experts (MoE)
    * Low-rank adaptation (LoRA)
  * 浮点数
    * FP32, FP16, BF16, FP8


---


# Mixer-of-Experts (MoE)

* 从分而治之的思想说起
  * 术业有专攻：将问题拆分为多个子问题，每个子问题有自己的专家
    * 如何备战高考：语文、数学、英语、物理、化学...
* 模型中的“专家”
  * 问题：用户输入
  * 子问题：用户输入涉及什么类型
  * 专家：针对子问题的模型
    * ensemble视角：多个“一般”的模型
  * 选择专家：通过gating model/network选择最适合的专家
---

# Example of mixture of experts

图片来源[Ensemble Methods](https://amzn.to/2XZzrjG)
![w:800 center](images/l8/expert_ensemble.png)


---

# NLP模型中的Mixture of experts


[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
* 设计了MoE layer
  * 特点：稀疏(sparsely)、快速推理

<div style="display:contents;" data-marpit-fragment>

![w:500 center](images/l8/17_sparse_moe_layer.png)

</div>

---

# MoE中的稀疏

![w:1000 center](images/l8/17_sparse_moe_layer.png)

---

# MoE中的稀疏

* 模型体积
  * 单expert尺寸为$s$
* 计算开销
  * 单expert的计算开销记为$c$
* 传统模型/单expert：体积$s$，计算开销$c$
* Sparsely MoE(M个expert)，通过Gating network，选择最适合的expert进行推理计算
  * 模型体积：$M$个expert的MoE layer体积$ms$
  * 计算开销：选择一个expert的情况下，计算开销为$c$

---

# MoE中的稀疏

不“怎么”增加模型计算开销的情况下，提升模型的体积
![w:1000 center](images/l8/17_sparse_moe_layer.png)

---

# MoE结构示例

* 一种"weighted multiplication"
  * 其中$G(x)$为gating network(可学习)，$E_i(x)$为expert
  * $y=\sum_{i=1}^nG(x)_iE_i(x)$

* $G(x)$通常为$\text{softmax}$，配以可学习的参数$W_g$
  * $G(x)=\text{softmax}(x\cdot W_g)$
* 为了体现稀疏性，一般选Top-k Gating
  * 通常　$k=1,2$
---

# MoE与Transformers

[Switch Transformers](https://arxiv.org/abs/2101.03961)将Transformers中的FFN替换为MoE结构
![w:900 center](images/l8/switch_transformer.png)

---

# MoE与Transformers

* [Switch Transformers](https://arxiv.org/abs/2101.03961)
  * T5-based MoEs going from 8 to 2048 experts
* [Mixtral 8x7B](https://huggingface.co/mistralai) 
  *  8 experts


---

# 如何确定MoE中的expert

* 通过训练过程学习参数，确定每个expert
* 以Mixtral 8x7B为例
  * 将FFN layer扩展为8个expert
    * 每个expert为原有尺寸的一个FFN layer
    * 通过Gating network选择最适合的expert


---


# 前馈神经网络(FFN)


![w:1100 center](images/l5/pipeline.png)


---

# 标准FFN实现

```python
(mlp): LlamaMLP(
  (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
  (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
  (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
  (act_fn): SiLU()
)
```
* 增加Gating network，forward function中通过Gating network选择最适合的expert，每个expert进行以下计算：
  * self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)

---

# MoE的训练

* 通过load balancing训练MoE
* MoE训练之前：初始化每个expert，且并无任何含义
* 不加任何控制的训练：每次选取top-k(=2)的expert进行训练和参数更新，容易使模型选择被训练的快的experts
* load balancing: 在训练过程中，赋予每个expert近乎相同数量的训练样本




---

# Low-rank adaptation (LoRA)

* 一种流行的轻量级LLM微调技术
  * 通过很少的trainable parameters,快速微调LLM
    * [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)


<div style="display:contents;" data-marpit-fragment>

![w:800 center](images/l8/lora.jpeg)

</div>

---

# LoRA基本思路

* 传统的training和finetuning
  * $W_{updated}=W_{old}+\Delta W$
    * $\Delta W$ size: [4096, 4096]
* LoRA
  * $\Delta W \approx AB$, rank $r$=8 (可调)
    * $A$ size: [4096, 8]
    * $B$ size: [8, 4096]
  * $W_{updated}\approx W_{old}+\alpha AB$
    * $\alpha$: scaling factor

---

# LoRA推理

* 传统的training和finetuning后的推理过程
  * $x\cdot W_{updated} = x\cdot (W_{old}+\Delta W)=x\cdot W_{old}+ x\cdot \Delta W$
* LoRA推理
  * $x\cdot W_{updated} \approx x\cdot (W_{old}+AB)=x\cdot W_{old}+ x\cdot \alpha AB$


---

# LoRA实现

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
```

---


# LoRA实现

```python
class LinearWithLoRA(nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```
移步notebook


---

# 浮点数表示

* 浮点数，又称floating point，是一种有符号的二进制数，考虑其精度(Precision)，通常有：
  * 双精度: FP64
  * 单精度: FP32
  * 半精度: FP16

<div style="display:contents;" data-marpit-fragment>

![w:1000 center](images/l8/fp32.png)

</div>

---

# 浮点数表示

* 以FP32为例:
  * 符号位: 1位 sign
  * exponent部分: 8位 exponent 
  * fraction部分: 23位 fraction
  * $value=(-1)^{\text{sign}}\times2^{E-127}\times(1+\sum_{i=1}^{23}b_{23-i}2^{-i})$

<div style="display:contents;" data-marpit-fragment>

![w:1000 center](images/l8/fp32.png)

</div>

---

# 浮点数表示

![w:1000 center](images/l8/bf16-fp32-fp16.png)

---

# 浮点数表示

![w:1000 center](images/l8/fp8_formats.png)

