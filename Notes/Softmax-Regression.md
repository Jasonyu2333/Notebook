## Softmax Regression 

### 概念

输出的是一个合法的类别预测分布，满足非负和归一性，输出值个数等于标签的类别数。Softmax是一个全连接的单层神经网路。

![img](../Images/3.4_softmaxreg.svg)

### 模型

##### 对于单个样本 i，输入的特征值有d个，输出值有q个，softmax回归的矢量计算表达式为：

${\boldsymbol{O}}^{(i)} = {\boldsymbol{x}}^{(i)}\boldsymbol{W} + \boldsymbol{b} $ （各矩阵的形状为：1\*q, 1\*d，d\*q，1\*q）

> 实现了d维到q维的线性空间变换。

${\hat{\boldsymbol{y}}}^{(i)} = softmax({\boldsymbol{O}}^{(i)})$ ，其中  $\hat{y_{1}} = P\left ( y = i \right ) = \frac{e^{O_{i}}}{\sum_{j=0}^{q-1}e^{O_{j}}}$

> 进行softmax运算，实现同维变换，输出一个合法的概率分布。

##### 对于小批量样本，输入的批量为n，softmax回归的矢量计算表达式为：

${\boldsymbol{O}} = {\boldsymbol{X}}\boldsymbol{W} + \boldsymbol{b}$ （各矩阵的形状为：n\*q, n\*d，d\*q，1\*q，加法用到广播机制）

${\hat{\boldsymbol{Y}}} = softmax({\boldsymbol{O}})$

### 损失函数

> ![image-20201028131138666](../Images/image-20201028131138666.png)

交叉熵（cross entropy）是一个常用的衡量两个概率分布差异的测量函数

$H({\boldsymbol{y}}^{(i)},\hat{\boldsymbol{y}}) = -\sum_{j=0}^{q-1}y_{j}^{(i)}\log {{\hat{y}}_{j}^{(i)}}$





