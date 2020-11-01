## Mini-Batch

### 概念

- epoch：训练轮数
- batch-size：一次训练的样本数
- teration：迭代次数

### 实现

 利用Dataset和Dataloader创建迭代器，由嵌套函数完成训练。

```python
# 每个循环完成一个训练轮数
for epoch in range(training_epochs):
	# 每个循环完成一个Mini-batch，train_loader由Dataloader创建
	for i, data in enumerate(train_loader, 0):
```

