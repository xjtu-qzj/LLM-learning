# PyTorch 基础机制

## 1. `torch.nn.Module` 是什么？

### 问题
为什么自定义模型、层和组件几乎都要继承 `torch.nn.Module`？

### 核心结论
`nn.Module` 是 PyTorch 神经网络系统的基础抽象。继承它之后，模块才能自动接入参数管理、设备迁移、训练推理模式切换和序列化等机制。

### 解释
`nn.Module` 提供的核心能力包括：
- 参数注册：`nn.Parameter` 会被自动纳入 `model.parameters()`。
- 子模块追踪：子层会被自动组织成模块树。
- 模式切换：支持 `train()` 和 `eval()`。
- 状态保存：支持 `state_dict()` 与 `load_state_dict()`。
- 设备迁移：支持 `to(device)`。

### 补充
从工程角度看，继承 `nn.Module` 的价值不只是“能写 `forward`”，而是完整接入 PyTorch 的模块生态。

---

## 2. Hooks 是什么？

### 问题
Hook 在 PyTorch 里到底解决什么问题？

### 核心结论
Hook 允许你在不修改模型主体代码的前提下，观察或干预前向、反向传播过程，是调试、解释模型和分析梯度的重要工具。

### 解释
#### Module Hooks
- `register_forward_hook`：前向传播执行后触发，适合提取中间层输出。
- `register_forward_pre_hook`：前向传播执行前触发，可检查或修改输入。
- `register_full_backward_hook`：反向传播相关阶段触发，适合分析梯度流。

#### Tensor Hooks
- `register_hook`：张量梯度产生时触发，可直接观测或变换梯度。

### 最小示意
```python
handle = layer.register_forward_hook(
    lambda module, inputs, output: print(output.shape)
)
```

### 补充
Hook 很适合做调试和分析，但如果用来修改计算路径，需要非常谨慎，否则容易引入难排查的副作用。

---

## 3. `super().__init__()` 的作用是什么？

### 问题
为什么自定义 `nn.Module` 时，通常第一行就要写 `super().__init__()`？

### 核心结论
这是为了执行父类 `nn.Module` 的初始化逻辑。没有这一步，模块系统的关键内部结构不会建立，参数和子模块也无法被正常管理。

### 解释
如果不调用：
- `self.linear = nn.Linear(...)` 这类子模块可能不会被正确注册。
- `parameters()` 可能拿不到完整参数。
- `state_dict()`、`to()`、`eval()` 等行为可能异常。

### 最小示意
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128)

    def forward(self, x):
        return self.proj(x)
```

### 补充
`super().__init__()` 不是语法礼仪，而是 `nn.Module` 正常工作的前提。
