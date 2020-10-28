## Softmax Regression 



### Softmax线性回归模型

输出的是一个概率分布，满足非负和归一性，输出值个数等于标签的类别数。Softmax是一个全连接的单层神经网路。

![img](../Images/3.4_softmaxreg.svg)

输入值有m个，输出值有n个，

对于单个样本 i，softmax回归的矢量计算表达式为：

$$
{\boldsymbol{O}}^{(i)} = {\boldsymbol{x}}^{(i)}\boldsymbol{W} + \boldsymbol{b}（各矩阵的形状为：n*1, m*1，m*n，n*1）
$$
实现了m维到n维的线性空间变换

${\hat{\boldsymbol{y}}}^{(i)} = softmax({\boldsymbol{O}}^{(i)})$



$ \hat{y_{1}} = P\left ( y = i \right ) = \frac{e^{O_{i}}}{\sum_{j=0}^{N-1}e^{O_{j}}} $
