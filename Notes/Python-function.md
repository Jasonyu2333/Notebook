##  enumerate() 函数

### 描述

用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。返回一个`enumerate`枚举对象。

### 语法

```python
enumerate(sequence, [start=0])
# sequence：迭代对象，start：下标起始位置
```

### 实例

```python
seq = ['one', 'two', 'three']
for i, data in enumerate(seq):
	print i, element
... 
0 one
1 two
2 three
```

