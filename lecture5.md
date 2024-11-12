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

第5讲: 大语言模型解析 II

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
* **Transformer部分**
  * **Attention module**
  * **Feed forward network**

---

# HF LlaMA模型结构

```python
LlamaForCausalLM(
  (model): LlamaModel(
    ...
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention
        (mlp): LlamaMLP
        (input_layernorm): LlamaRMSNorm
        (post_attention_layernorm): LlamaRMSNorm
    )
    ...
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
```

![bg right:30% 100%](images/l4/llama_arch_rope.png)




---

# LlamaDecoderLayer内部结构

```python
(self_attn): LlamaAttention(
  (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
  (k_proj): Linear(in_features=2048, out_features=512, bias=False)
  (v_proj): Linear(in_features=2048, out_features=512, bias=False)
  (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
  (rotary_emb): LlamaRotaryEmbedding()
)
(mlp): LlamaMLP(
  (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
  (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
  (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
  (act_fn): SiLU()
)
(input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
(post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
```

---

# LlamaDecoderLayer内部结构

* Normalization模块
  * LlamaRMSNorm
* 两个主要模块
  * LlamaAttention
  * LlamaMLP


---

# Normalization

```python
(input_layernorm): LlamaRMSNorm
```

* Normalization，中文又称“归一化”，“标准化”,... 
* 作用：调整数据的分布(scaling)，亦可称“缩放”
*  数据包括：输入、输出、中间层表示，...

<div style="display:contents;" data-marpit-fragment>

最经典的Normalization

Standard score: $\frac{X-\mu}{\sigma}$, Minmax feature scaling: $\frac{X-X_{\min}}{X_\max-X_{\min}}$

</div>

---

# Normalization示例

<div style="display:contents;" data-marpit-fragment>


一维的输入，归一化后的输出
![w:700 center](images/l5/animation-normalizing-1.gif)

</div>

---

# Normalization在机器学习中的应用

“药效”：加速训练收敛，让输入更“规整”，降低过拟合(overfitting)，增强泛化(generalization)

![w:900 center](images/l5/normal_for_training.png)


---


# Normalization v.s. Regularization

* 目标不同：
  * Normalization=调整数据
    * 比如: $X'=X-\frac{X_\min}{X_\max-X_\min}$
  * Regularization=调整预测/损失函数
    * 比如: $\text{loss}=\min\sum_{i=1}^N L(f(x_i), y_i)+\lambda R(\theta_f)$


---


# 大语言模型引入Normalization

Normalization：“调整数据分布”

* 数据长什么样子？都是tensor！维度很多！
  * 原始输入: vocab embedding
    * tensor shape: <batch_size, sequence_length, hidden_dim>
  * 深度学习模型中间层表示(hidden states/representations)
      * tensor shape: <batch_size, sequence_length, hidden_dim>

---

# 大语言模型引入Normalization



<div style="display:contents;" data-marpit-fragment>

![w:900 center](images/l5/normalization.png)

</div>

---

# Normalization的设计思路

* tensor shape: <batch_size, sequence_length, hidden_dim>
* 选择最合适的Normalization维度
  * batch：X=[batch_size,sequence_length, hidden_dim]
  * sequence： X=[sequence_length, hidden_dim]
  * hidden: <bs, seq, hidden> => <N, hidden>, X=[hidden_dim]
    * 又称LayerNrom

<div style="display:contents;" data-marpit-fragment>

当前流行的LayerNorm：[RMSNorm](https://arxiv.org/pdf/1910.07467)

</div>

---

# RMSNorm

* torch 2.4，提供了RMSNorm类的实现[torch.nn.RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html#torch.nn.RMSNorm)
* 原理: "regularizes the summed inputs to a neuron in one layer according to root mean square (RMS)"
  * $y = \text{RMSNorm}(x)=\frac{x}{\sqrt{\text{RMS}[x]+\epsilon}}*\gamma$
  * $\text{RMS}(x)=\sqrt{\frac{\sum_{i=1}^N x_i^2}{N}}$

<div style="display:contents;" data-marpit-fragment>

如何利用torch算子自行实现RMSNorm？

</div>

---

# 手搓RMSNorm

$y = \text{RMSNorm}(x)=\frac{x}{\sqrt{\text{RMS}[x]+\epsilon}}*\gamma$， $\text{RMS}(x)=\sqrt{\frac{\sum_{i=1}^N x_i^2}{N}}$

* 输入和输出的shape: <batch_size, sequence_length, hidden_dim>
* 涉及的计算
  * 平方，求和，开根
  * torch提供: tensor.pow, tensor.mean, torch.rsqrt

<div style="display:contents;" data-marpit-fragment>

我们来试试

</div>

---

# RoPE实现

RoPE的2D实现
![w:800 center](images/l4/roformer_fqk.png)

RoPE的n维实现
![w:800 center](images/l4/roformer_nd.png)


---

# RoPE实现

* 目标：实现RoPE(对应的$R_{\Theta, m}^d$)
  * 准备一堆$\theta_i$，以及对应的cos和sin数值

* 构建RoPE矩阵: $R_{\Theta, m}^d$流程
  * 批量算cos，再批量算sin
  * 涉及torch算子
    * torch.arrange, torch.sin, torch.cos, torch.outer

<div style="display:contents;" data-marpit-fragment>

我们来试试

</div>

---

# 前馈神经网络(FFN)


![w:1100 center](images/l5/pipeline.png)


---

# FFN实现，LlamaMLP

```python
(mlp): LlamaMLP(
  (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
  (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
  (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
  (act_fn): SiLU()
)
```

* 组件：三个nn.Linear层，一个SiLU激活函数
  * SiLU: torch.nn.functional.silu(x)
  * Linear: torch.nn.Linear(in_features, out_features)


---

# FFN流程

* 输入tensor: <batch_size, sequence_length, hidden_dim>
* 第一步：
  * 通过gate_proj获得gate tensor，经过SiLU激活得到gate tensor
  * 通过up_proj获得up tensor
* 第二步：元素乘(elementwise multiply): gate tensor 和 up tensor
* 第三步: 通过down_proj获得down tensor

<div style="display:contents;" data-marpit-fragment>

摘抄自transformers/src/models/modeling_llama.py
```python
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)
```

</div>