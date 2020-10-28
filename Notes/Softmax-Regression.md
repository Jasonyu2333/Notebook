## Softmax Regression 



### Softmax线性回归模型

输出的是一个概率分布，满足非负和归一性，输出值个数等于标签类别数。Softmax是一个全连接的单层神经网路。

![img](../Images/3.4_softmaxreg.svg)

输出层有N个输出$0_{1}$,$0_{2}$...

$ {y}_{i} = P\left ( y = i \right ) = \frac{e^{O_{i}}}{\sum_{j=0}^{N-1}e^{O_{j}}} $

$\frac{ew}{233}$