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

第9讲: 大语言模型解析 VI

基于HF LlaMA实现的讲解

<!-- https://marp.app/ -->

---

# LLM结构的学习路径

* LLM结构解析(开源LlaMA)
* 自定义数据集构造
* 自定义损失函数和模型训练/微调

---

# 如何让LLM“动”起来

<!-- ![bg right:40% 100%](images/l4/transformer.png) -->

<!-- ![bg right:30% 100%](images/l4/llama_arch_rope.png) -->

* 训练
  * 预训练 (pretraining)
    * 继续预训练(Continuous PreTraining, CPT)
  * 指令微调 (INstruction fine-tuning)
    * 监督微调 (Supervised Finetuning, SFT)
    * RLHF (带人类反馈(Human feedback)的强化学习(RL))
* 推理

---

# 数据集

* 预备 ```pip install datasets```
* 人类视角下的数据集 v.s. LLM视角下的数据集
  * 转换工具: tokenizer
<div style="display:contents;" data-marpit-fragment>

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

</div>

* 通过tokenizer将原始文本编码(encode)为token序列
<div style="display:contents;" data-marpit-fragment>


```python
encoded_input = tokenizer("Tell me a story about Nanjing University.")
```

</div>

---

# token序列 <---> 文本

* 字典结构
  * input_ids: token id
  * attention_mask
* 操作
  * encode
  * decode
  * padding
  * truncation


---

# 字典结构

基本元素：input_ids 和 attention_mask
```python
encoded_input = tokenizer("Tell me a story about Nanjing University.")
```
通过tokenizer编码后的token序列
```python
{
  'input_ids': [41551, 757, 264, 3446, 922, 33242, 99268, 3907, 13], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```


---

# 编码和解码

* 编码(encode)
  * tokenizer(input): 得到{'input_ids','attention_mask'}字典结构
  * tokenizer.tokenize(input): 得到tokens
  * tokenizer.encode(input): 得到tokens的ids
* 解码(decode)
  * tokenizer.decode(input): 得到文本
    * input为ids的List


---

# 如何批处理

* 多段文本组成的batch
<div style="display:contents;" data-marpit-fragment>

```python
batch_sentences = [
    "Tell me a story about Nanjing University.",
    "耿鬼能超进化么？",
    "大语言模型课程怎么考试？",
]
encoded_inputs = tokenizer(batch_sentences)
```

</div>

---

# 如何批处理

输出结果

```python
{'input_ids': 
  [[41551, 757, 264, 3446, 922, 33242, 99268, 3907, 13], 
  [20551, 123, 111188, 27327, 72404, 42399, 33208, 82696, 11571], 
  [27384, 120074, 123123, 123440, 104237, 118660, 11571]], 
'attention_mask': 
  [[1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1]]
}
```


---

# 批处理内容不一样长

```python
batch_sentences = [
    "Tell me a story about Nanjing University.",
    "耿鬼能超进化么？",
    "大语言模型课程怎么考试？",
]
```

添加padding

```python
encoded_input = tokenizer(batch_sentences, padding=True)
```

---

# Padding

```python
{'input_ids': 
  [[41551, 757, 264, 3446, 922, 33242, 99268, 3907, 13], 
  [20551, 123, 111188, 27327, 72404, 42399, 33208, 82696, 11571], 
  [27384, 120074, 123123, 123440, 104237, 118660, 11571, 128009, 128009]], 
'attention_mask': 
  [[1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1, 0, 0]]
}
```

---

# Padding


* 指定长度进行padding

<div style="display:contents;" data-marpit-fragment>
  
```python
encoded_input = tokenizer(batch_sentences, padding="max_length", max_length=20, truncation=True)
```

</div>

* 控制padding方向: padding_side
  * tokenizer.padding_side: left or right
<div style="display:contents;" data-marpit-fragment>

```python
tokenizer.padding_side = 'left'
encoded_input = tokenizer(batch_sentences, padding="max_length", max_length=20, truncation=True)
```

</div>

---

# 其他

* 句子太长，LLM无法处理
  * 指定长度进行truncation
    * 调用tokenizer时配置参数```truncation=True```
* 将token序列转化为tensor格式
  * 调用tokenizer时配置参数```return_tensors="pt"```


---

# 加载数据集

```python
from datasets import load_dataset

ds = load_dataset("yahma/alpaca-cleaned")
```

* 数据集有其自身格式，一般地，包含'train', 'validation', 'test'部分
  * 调用```load_dataset()```方法后获得数据集字典
    * 获取训练集```ds['train']```
    * 看看数据集构成...


---

# 加载数据集

* 需实现数据集的预处理方法，并交由Datasets的map方法调用
  * 预处理方法
<div style="display:contents;" data-marpit-fragment>

```python
def tokenize_function(dataset):
  ...
  return ...
  ```

</div>

* 调用预处理方法
<div style="display:contents;" data-marpit-fragment>

```python
ds = load_dataset("yahma/alpaca-cleaned", split='train[:100]')
ds = ds.map(tokenize_function, batched=True)
```
</div>


---

# 微调模型

加载模型
```python
model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)
```
设置训练参数
```python
from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="test_trainer")
```

---

# 微调模型

移步vscode

---

# 阅读HF LlaMA实现

移步vscode