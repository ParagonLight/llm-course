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




![bg left:45% 80%](images/course.webp)

# **LLM智能应用开发**

第1讲: 课程简介

<!-- https://marp.app/ -->

---

# 教师信息

讲师: 徐经纬

办公室: 计算机学院1022

邮箱: jingweix@nju.edu.cn

课程主页: TBD

---

# 课程基本信息

![bg right:50% 80%](images/l1/qq.jpg)

- 课程大纲
- 参考资料
- QQ群: 208283743
- 分数组成
  * 平时编程作业 60%
  * 期末编程大作业 40%

* 注: 对GPU资源有一定要求

---



# 课程前身: iOS智能应用开发
- Swift程序设计语言
- iOS App开发技术
- iOS 人工智能（机器学习）技术应用
  - 图像类应用
  - 文本类应用

---

# 课程大纲: LLM应用架构图


![w:800 center](images/l1/syllabus.svg)

---

# 课程大纲: 授课脉络

![w:700 center](images/l1/syllabus_1.jpg)

---

# 深度学习基础

* 何为深度(机器)学习模型
* 如何开发一个模型
  * 数据、模型、训练算法
  * PyTorch基础*
* 相关概念光速入门
  * 机器学习概念、反向传播

---

# 语言模型核心基础

* 一切源自于如何建模自然语言: 统计模型
  * Bigram/N-gram Language Model
* 模型如何认识单词: 问就是编码
  * one-hot编码, word2vec, tokenization
* 模型如何理解语言: Attention is All You Need
  * attention, softmax, positional embedding, ... 

---

# LLM经典架构解析

* 了解基础组件
  * Transformer, 残差(residual), layernorm
* 一起搭LLM积木
  * Encoder-decoder, decoder only
* 代码案例: 典型LLM架构代码解析
  * LlaMA家族

---

# LLM经典架构解析(续集)

“虚的讲完了，让我们实际一点”
* LLM如何在计算机中被创建
* 运行设备：CPU，GPU，TPU...
* 精度: fp32, fp16, bf16, 混合精度...
* 分布式运行: 分布式数据并行DDP, ZeRO, FSDP, Deepspeed...
  
---

# LLM训练和推理

* LLM预训练和微调训练
  * 训练概念介绍
  * 数据集和数据加载过程构建
  * 微调流程构建(含对齐(Alignment))
   * SFT: SFT, PEFT, LoRA
    * RL*: RLHF, PPO, DPO
* 如何推理模型
  * KV-cache, 量化(Quantization)

---

# LLM应用

应用技术
* 检索增强生成(RAG)
* LLMs as Agents
  
应用场景
* 聊天, 多模态, 数学, 代码生成

---

# 这是一门理论基础和实践相结合的课程

* 实践体现在无处不在的手撸代码过程
* 你会了解和学习:
  * PyTorch
  * Transformers and PEFT (from Huggingface)
  * 以及其他流行开源框架

<div style="display:contents;" data-marpit-fragment>
例如: 以下代码是如何执行的？

```python
model.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
```

</div>

---

![bg](images/l1/chatgpt.jpg?text=ChatGPT)

---

# 人工智能是


人工智能=人工+智能

训练过程
人工：大力出奇迹
智能：构建分布

推理过程
条件概率 $\mathrm{p}(y|x)$

![bg right:63% 100%](images/l1/generation.jpg)

---

# 什么是自然语言处理？



* 自然语言
  * 人类使用的语言，如汉语、英语、西班牙语、代码等 -> 文本符号
* 自然语言处理的定义
 * “自然语言处理（NLP）是语言学、计算机科学和人工智能的跨学科子领域，关注计算机和人类语言之间的交互，特别是如何编程使计算机能够处理和分析大量的自然语言数据。其目标是使计算机能够“理解”文档的内容，包括其中的语言背景细微差别。然后，这项技术可以准确提取文档中包含的信息和见解，以及对文档本身进行分类和组织。” *From WikiPedia, ChatGPT翻译*
 * <span style="color:red;">自然语言理解，自然语言生成</span>


---

# 自然语言处理任务

* 请填空: 今天我来到了___
* 今天的天气____
* 自然语言处理的任务本质上是一个“填词游戏”
  * 目标：如何填的准、说得好
  * 手段：条件概率$\mathrm{p}(y|x)$
  * 填的准≠人类知道机器按照人类理解世界的方式理解世界


---

# 语言模型

![bg right:40% 80%](images/l1/distribution.jpg)

基本法: 链式法则

句子由任意长度的字符串组成

- 句子a = 今天我来到了仙II-212。
- 句子b = 今天仙II-212我来到了。
* 用概率衡量句子的“好”: $\mathrm{p}(a) > \mathrm{p}(b)$
*自然语言处理(NLP)模型：估计出的(相对准确的)概率分布

---

# 语言模型

基本法: 链式法则

* 理想: 直接估计出的(相对准确的)句子概率分布
* 现实: 参考人类小孩学说话，一个字一个字的说
  * 怎么学? 链式法则

<div style="display:contents;" data-marpit-fragment>

$\mathrm{p}$(今天我上智能应用开发。) = $\mathrm{p}$(今) $\mathrm{p}$(天|今) $\mathrm{p}$(我|今天) $\mathrm{p}$(上|今天我)...
$\mathrm{p}$(。|今天我上智能应用开发)

</div>

<div style="display:contents;" data-marpit-fragment>

<span style="color:red;">这就是经典的N-gram model</span>

</div>

---

# LLM的前寒武纪时期

![bg right:43% 100%](images/l1/transformer.png)

* 2017年6月之前
  * RNN系列
* 2017年-2022年
  * Attention
  * Transformer界的剑宗气宗之争
    * Encoder-decoder派: BERT
    * Decoder-only派: GPT系列

---


# Transformer界的剑宗气宗之争

文字接龙(GPT) v.s. 完形填空(BERT)

<style>
img[alt~="top-right"] {
  display: block;
  margin: 0 auto;
}
</style>

<style>
img[alt~="bottom-right"] {
  position: absolute;
  top: 400px;
  right: 0px;
}
</style>

<!-- ![top-right](images/gpt.png) -->

<p align="center">
  <img width="380" height="400" src="images/l1/gpt.png">
  <img width="450" height="400" src="images/l1/bert.png">
</p>

<!-- ![w:400 h:400](images/gpt.png)  ![w:320 h:320](images/bert.png) -->

---

# LLM的寒武纪大爆发

- OpenAI发布ChatGPT
  * GPT系列: GPT-3.5, GPT-4, GPT-4 Turbo, GPT4o...
* 其他公司
  * 国外: LlaMA家族, Gemini, Mistral, Mixtral, Claude, ...
  * 国内: Deepseek, 文心一言, GLM(智谱清言), Moonshot(月之暗面), 通义千问, abab(MiniMax), ...

---

![bg 60%](images/l1/llm_tree.png)
<!-- <p align="center">
  <img width="500" height="500" src="images/llm_tree.png">
</p> -->
---

![bg 90%](images/l1/llama_family.png)
<!-- <p align="center">
  <img width="500" height="500" src="images/llm_tree.png">
</p> -->
---

# 开发LLM少不了开源社区的支撑

* 深度学习框架: PyTorch
* 模型社区: Huggingface
* 其他: LlaMA-factory, deepspeed, magatron, triton, llama.cpp, llama2.c, llm.c, ...

* 开发语言: Python, CUDA, C++, ...

---

# Talk is cheap. Show me the code.

准备:

* Python 3.8+
* 设备: 最好有N卡，实在没有也没关系，内存大一点
* 虚拟环境: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
* 环境配置: PyTorch, Transformers, PEFT, triton, ...
  * 绝大部分可通过pip或conda安装

<!-- ![bg](https://fakeimg.pl/800x600/02669d/fff/?text=Show+me+the+code) -->