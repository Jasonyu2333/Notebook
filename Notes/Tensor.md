## WHAT is Tensor? 

torch.Tensor是存储和变换数据的主要工具，可以看作单一多维矩阵。

## WHY is Tensor?

Tensor提供GPU计算和自动求梯度等更多功能，更适合深度学习。

#### view(*args)

返回一个有相同数据但大小不同的tensor，参数中允许有一个-1，系统将自动计算该值，使得参数连乘等于原Tensor元素相同。