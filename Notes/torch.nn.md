# torch.nn

## torch.nn.Module

所有网络的基类，所有模型都应继承这个类。

```python
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
```

## Linear layers

```python
torch.nn.Linear(in_features, out_features, bias=True)
```

#### 参数

- in_features：输入样本的维度，输入的矩阵为（N，in_features）
- out_features：输出样本的维度，输出的矩阵为（N，out_features）
- bias：若设置为False，这层不会学习偏置。

