```python
# 验证Pytorch是否使用GPU
import torch
from torch.backends import cudnn
# 测试CUDA
print("Support CUDA ?: ", torch.cuda.is_available())
x = torch.tensor([10.0])
x = x.cuda()
print(x)
y = torch.randn(2, 3)
y = y.cuda()
print(y)
z = x + y
print(z)
# 测试CUDNN
print("Support cudnn ?: ", cudnn.is_acceptable(x))
```