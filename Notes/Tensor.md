### Tensor（张量）

## WHAT is Tensor? 

Tensor是存储和变换数据的主要工具，可以看作多维矩阵。

## WHY Tensor?

Tensor提供GPU计算和自动求梯度等更多功能，更适合深度学习。

## 创建Tensor

| 函数                              | 功能                      |
| :-------------------------------- | :------------------------ |
| Tensor(*sizes)                    | 基础构造函数              |
| tensor(data,)                     | 类似np.array的构造函数    |
| ones(*sizes)                      | 全1Tensor                 |
| zeros(*sizes)                     | 全0Tensor                 |
| eye(*sizes)                       | 对角线为1，其他为0        |
| arange(s,e,step                   | 从s到e，步长为step        |
| linspace(s,e,steps)               | 从s到e，均匀切分成steps份 |
| rand/randn(*sizes)                | 均匀/标准分布             |
| normal(mean,std)/uniform(from,to) | 正态分布/均匀分布         |
| randperm(m)                       | 随机排列                  |





#### view(*args)

返回一个有相同数据但大小不同的tensor，参数中允许有一个-1，系统将自动计算该值，使得参数连乘等于原Tensor元素相同。