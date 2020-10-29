# Optimization-algorithms(优化算法)

## WHAT is Optimization-algorithms？

找到一组参数使得损失函数最小的算法。

## WHY Optimization-algorithms？

目标函数（Loss function）是没有解析解的，无法直接得到最小值。所以就需要优化算法通过迭代运算尽可能减少损失函数的值，迭代的过程中会遇到两个问题：

- 局部最优解

    对于非凸函数，存在局部最优解和全局最优解的问题，幸运地是在高维空间内局部最优解并不多。

- 鞍点

    g = 0，无法迭代，是核心要处理的问题。

## Implementation

生成优化器调用函数

```python
optimizer = optim.***()
```

进行单次优化（需在backward等函数计算后）

```python
optimizer.step()
```

## Types of Optimization-algorithms

### SGD（随机梯度下降）

```python
torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

#### 参数：

- params：待优化的可迭代的对象(iterable)或者是定义了参数组的dict
- lr (`float`)：学习率
- momentum (`float`, 可选)：动量因子
- weight_decay (`float`, 可选)：权重衰减（L2惩罚）
- dampening (`float`, 可选)：动量的抑制因子
- nesterov (`bool`, 可选)：使用Nesterov动量

#### Tip：

​	梯度下降