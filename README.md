# 基于BERT的情感分析学习项目

## 项目概述

这是一个用于学习NLP和深度学习的实践项目，通过微调BERT预训练模型来实现电影评论的情感分类（正面/负面）。项目展示了如何使用Hugging Face Transformers库完成一个完整的文本分类任务，从数据准备到模型部署的全过程。

> 本项目主要用于学习目的，使用了极小的示例数据集来演示流程，实际应用需要更大规模的数据集。因电脑性能不够,我这里用的模型很小且对中文不太友好读者可以自行更换需要的模型

源码[BERT文本分类](https://github.com/yangdongsheng02/HZ_AI/blob/master/%E4%BA%A7%E5%87%BA%E9%A1%B9%E7%9B%AE%E9%9B%86%E5%90%88/BERT%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.py)
##  项目目标

使用BERT预训练模型构建了一个电影评论情感分析系统。系统能够自动判断电影评论的情感倾向（正面/负面），并给出置信度评分。

## 技术栈

| 技术                | 用途        |
| ----------------- | --------- |
| **Python 3.8+**   | 编程语言      |
| **PyTorch**       | 深度学习框架    |
| **Transformers**  | BERT模型和工具 |
| **Pandas**        | 数据处理      |
| **scikit-learn**  | 数据集划分     |


## 项目结构

```
本项目采用模块化设计，包含以下核心文件和目录：
BERT.py - 主训练脚本，包含数据准备、模型训练和评估的完整流程
    
results - 训练结果（运行后生成）
    
my_sentiment_model/ - 训练好的模型保存目录（运行后自动生成），包含模型权重、配置和分词器文件
    
```

##  快速开始

### 1. 环境配置

```bash

# 安装依赖（推荐使用虚拟环境）
import os  
import pandas as pd  
from sklearn.model_selection import train_test_split  
import accelerate
```

### 2. 运行训练

```bash
# 运行训练脚本
python BERT文本分类.py
```

### 3. 使用训练好的模型

```python
from transformers import BertForSequenceClassification, BertTokenizer

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('./my_sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./my_sentiment_model')

# 进行预测
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return "正面" if prediction == 1 else "负面"
```

## 核心代码解析

### 1. 数据准备

```python
# 创建示例数据集
data = {
    'text': [
        "This movie is absolutely fantastic!",
        "Terrible movie, waste of time.",
        # ... 
    ],
    'label': [1, 0, ...]  # 1=正面, 0=负面
}
df = pd.DataFrame(data)
```

### 2. 模型加载

```python
# 加载预训练的BERT模型和分词器
from transformers import BertTokenizer, BertForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

### 3. 文本预处理

```python
# 将文本转换为BERT能理解的格式
encodings = tokenizer(
    texts,
    truncation=True,     # 截断超长文本
    padding=True,        # 填充短文本
    max_length=128,      # 最大长度
    return_tensors="pt"  # 返回PyTorch张量
)
```

### 4. 自定义数据集

```python
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
```

### 5. 训练配置

```python
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=8,   # 批次大小
    eval_strategy="epoch",           # 评估策略
    logging_steps=10,                # 日志记录频率
    weight_decay=0.01,               # 权重衰减
)
```

## 模型限制

1. **序列长度限制**：最大512个token
2. **计算资源需求**：训练需要GPU支持
3. **领域适应性**：在专业领域可能需要领域自适应
4. **语言限制**：主要针对英文，其他语言需要相应模型

## 学习收获

### 理论知识方面
1. 理解了Transformer架构和自注意力机制
2. 掌握了BERT模型的工作原理

### 实践技能方面
1. 学会了使用Hugging Face Transformers库
2. 掌握了PyTorch数据加载和模型训练
3. 理解了文本分类的完整流程
4. 学会了模型保存和加载


##  问题与解决

### :运行时显存不足
**解决**: 
- 减小`batch_size`（如从16减到8）
- 减小`max_length`（如从256减到128）
- 使用梯度累积 `gradient_accumulation_steps=2`

###  训练损失不下降
**解决**:
- 检查学习率是否合适（BERT常用2e-5）
- 确保数据标签正确
- 增加训练轮数

###  模型预测结果总是同一类
**解决**:
- 验证模型是否保存/加载正确
- 确保预测时调用`model.eval()`

###  中文文本处理
**解决**:
- 使用中文预训练模型：`bert-base-chinese`
- 调整分词器：`BertTokenizer.from_pretrained('bert-base-chinese')`

## 项目可扩展
1. 使用真实数据集
2. 增加评估指标
3. 构建Web界面
4. 部署为API服务


##  作者

- GitHub: [@yangdongsheng02](https://github.com/yangdongsheng02)
- 邮箱: [1151717137@qq.com]
- 博客: [杨东升02.com](https://blog.csdn.net/m0_71469100)


---




