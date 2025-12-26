# transformers: Hugging Face的Transformer库，包含BERT等预训练模型
# torch: PyTorch深度学习框架
# datasets: Hugging Face的数据集处理库
# pandas, numpy, scikit-learn: 数据处理和机器学习常用库
import os
import pandas as pd
from sklearn.model_selection import train_test_split
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#数据准备
#创建数据集
# 这里使用一个简单的电影评论情感分析数据集
data = {'text': [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible movie, waste of time. Poor acting and weak plot.",
        "It was okay, not great but not bad either.",
        "One of the best films I've seen this year. Highly recommended!",
        "Disappointing. Expected much more from this director.",
        "A masterpiece of cinema. Brilliant performances all around.",
        "Boring and predictable. Fell asleep halfway through.",
        "Interesting concept but poor execution.",
        "Heartwarming story with great character development.",
        "Confusing plot with too many unnecessary subplots."
    ],
    'label': [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # 1=正面, 0=负面
}
# 创建DataFram
# 使用 pd.DataFrame() 函数从字典创建DataFrame
df = pd.DataFrame(data)     #每行评论对应一个label标签是正面还是负面
# 现在 df 就是一个包含两列('text'和'label')和三行数据的表格

# 划分训练集和验证集
# test_size=0.2表示20%的数据作为验证集
# random_state=42确保每次划分结果一致
#train_texts训练评论集   val_texts验证评论集
train_texts,val_texts,train_labels,val_labels = train_test_split(df['text'],df['label'],test_size=0.2,random_state=42)

#打印数据集大小
print("训练集大小:",len(train_texts))
print("验证集大小",len(val_texts))

#加载预训练模型和分词器
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# 指定使用的BERT模型名称
# "bert-base-uncased"是基础版本的BERT模型，使用小写字母

model_name = "bert-base-uncased"

#加载BERT分词器
#分词器负责将文本转换为模型可以理解的token ID
tokenizer = BertTokenizer.from_pretrained(model_name)

#加载BERT模型用于序列分类任务
#num_labels=2表示二分类任务(正面/负面)
model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2)

# 打印模型结构，了解模型的组成部分
print(model)

#数据预处理
#对训练集文本进行分词和编码
train_encodings = tokenizer(
    train_texts.tolist(), #将pandas Series转换为列表
    truncation=True,      #截断超过最大长度的文本
    padding=True,         #填充短于最大长度的文本
    max_length=128,       #设置最大序列长度
    return_tensors="pt"   #返回PyTorch张量
)

#对验证集文本进行分词和编码
val_encodings = tokenizer(
    val_texts.tolist(),  # 将pandas Series转换为列表
    truncation=True,  # 截断超过最大长度的文本
    padding=True,    # 填充短于最大长度的文本
    max_length=128,   # 设置最大序列长度
    return_tensors="pt"
)

#创建自定义PyTorch数据集类
import torch

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # 存储编码后的文本和标签
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self,idx):
    # 获取单个样本
    # 将编码数据转换为张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    #添加标签
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item
    def __len__(self):
    #返回数据集大小
        return len(self.labels)

#创建训练和验证数据集实例
train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 模型和检查点保存目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=8,   # 每个设备的训练批次大小
    per_device_eval_batch_size=16,   # 每个设备的评估批次大小
    warmup_steps=500,                # 学习率预热步数
    weight_decay=0.01,               # 权重衰减，防止过拟合
    logging_dir='./logs',            # 日志保存目录
    logging_steps=10,                # 每10步记录一次日志
    eval_strategy="epoch",     # 每个epoch结束后进行评估
)

# 创建Trainer实例，简化训练过程
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=val_dataset,            # 评估数据集
)

# 开始训练
trainer.train()


# 定义预测函数
def predict(text):
    # 对输入文本进行编码
    inputs = tokenizer(
        text,
        return_tensors="pt",  # 返回PyTorch张量
        truncation=True,  # 截断超长文本
        padding=True,  # 填充短文本
        max_length=128  # 最大序列长度
    )

    # 将模型设置为评估模式
    # 这会关闭dropout等训练特有的层
    model.eval()

    # 进行预测，不计算梯度以节省内存
    with torch.no_grad():
        outputs = model(**inputs)  # 将输入传递给模型

    # 获取模型的输出
    logits = outputs.logits  # 模型的原始输出（未经过softmax）

    # 应用softmax函数将logits转换为概率
    probabilities = torch.softmax(logits, dim=1)

    # 获取预测类别（概率最高的类别）
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # 返回预测结果和概率
    return predicted_class_id, probabilities[0].tolist()


# 测试几个例子
test_texts = [
    "This movie is great!",
    "I didn't like this film at all.",
    "It was an average movie, nothing special."
]

# 对每个测试文本进行预测
for text in test_texts:
    # 调用预测函数
    pred, probs = predict(text)

    # 将数字标签转换为可读的情感标签
    sentiment = "正面" if pred == 1 else "负面"

    # 打印预测结果
    print(f"文本: {text}")
    print(f"情感: {sentiment} (置信度: {max(probs):.4f})")
    print(f"负面概率: {probs[0]:.4f}, 正面概率: {probs[1]:.4f}")
    print("-" * 50)

    # 保存训练好的模型
model.save_pretrained("./my_sentiment_model")

    # 保存分词器
tokenizer.save_pretrained("./my_sentiment_model")

    # 加载模型
from transformers import BertForSequenceClassification, BertTokenizer

    # 从保存的目录加载模型
loaded_model = BertForSequenceClassification.from_pretrained("./my_sentiment_model")

    # 加载分词器
loaded_tokenizer = BertTokenizer.from_pretrained("./my_sentiment_model")