## Dataset

### 概念

用来创建数据集，它是一个抽象类，需要构造子类才能实例化，定义自己的继承和重写方法。

### 实现

引用包

```
import torch.utils.data.dataset as Dataset
import numpy as np
```

 创建dataset的子类

```
class subDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label
```

在类外对Data和Label赋值



## DataLoader

### 概念

DataLoader是一个迭代器，方便多线程的读取数据，主要用于实现batch以及shuffle。

### 实现

引用

```python
import torch.utils.data.dataloader as DataLoader
```

创建DataLoader迭代器

```python
dataloader = DataLoader.DataLoader(dataset,batch_size= 32, shuffle = True, num_workers= 2)
# dataset：传入数据集，batch_size：一次训练的样本数，shuffle：是否打乱，进程数
```

