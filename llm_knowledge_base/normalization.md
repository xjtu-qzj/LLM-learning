# Normalization

## 1. 为什么 Transformer 更常用 LayerNorm，而不是 BatchNorm？

### 问题
为什么在 Transformer 中通常使用 LayerNorm，而不是计算机视觉中常见的 BatchNorm？

### 核心结论
Transformer 更偏向使用 LayerNorm，因为它对单样本做归一化，不依赖 batch 统计量，更适合变长序列、padding 场景和 batch size 波动较大的训练过程。

### 公式

#### BatchNorm
对一个 batch 中某一维特征做归一化：

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

#### LayerNorm
对单个样本内部的隐藏维度做归一化：

$$
\mu = \frac{1}{d} \sum_{j=1}^{d} x_j
$$

$$
\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2
$$

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_j = \gamma_j \hat{x}_j + \beta_j
$$

### 解释
- **变长文本问题**：NLP 中同一个 batch 内的序列长度通常不同，padding 会污染 BatchNorm 的统计量。
- **样本内语义更关键**：语言建模更关注单个样本内部的表示结构，LayerNorm 的归一化维度更贴近这一点。
- **训练和推理更一致**：BatchNorm 训练时依赖 batch 统计量，推理时依赖滑动平均；LayerNorm 则没有这类切换成本。

### 补充
在 Transformer 中，归一化通常作用在 token 的 hidden dimension 上，而不是跨样本维度。

---

## 2. 归一化的作用是什么？

### 问题
归一化到底改善了什么，为什么几乎所有深层网络都离不开它？

### 核心结论
归一化的核心作用是稳定激活值尺度与梯度传播，使深层网络更容易优化，而不仅仅是把数据“变成标准正态”。

### 公式
一个通用归一化形式可以写成：

$$
\hat{x} = \frac{x - \text{center}(x)}{\text{scale}(x) + \epsilon}
$$

再经过可学习仿射变换：

$$
y = \gamma \hat{x} + \beta
$$

其中：
- `center(x)` 可能是均值，也可能省略。
- `scale(x)` 可以是标准差，也可以是 RMS。

### 解释
- **稳定数值分布**：避免层数增加后激活值不断放大或缩小。
- **改善梯度传播**：让输入更常落在激活函数较敏感的区域。
- **缓解梯度爆炸与消失**：使深层堆叠时训练更稳定。
- **优化更容易**：参数更新时对学习率和初始化通常没那么敏感。

### 补充
归一化不意味着每层输出一定严格服从标准正态分布，因为后面还有 `\gamma` 和 `\beta` 进行重新缩放和平移。

---

## 3. LayerNorm 中的 `γ` 和 `β` 分别学什么？

### 问题
既然已经做了标准化，为什么还要额外学习 `γ` 和 `β`？

### 核心结论
`γ` 控制幅值，`β` 控制偏移。它们让模型在获得归一化稳定性的同时，不失去表达能力。

### 公式
LayerNorm 的输出：

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

其中：
- `\gamma_i`：第 `i` 个维度的缩放参数
- `\beta_i`：第 `i` 个维度的偏移参数

### 解释
- `γ` 学习“这一维该放大多少”。
- `β` 学习“这一维的中心应该往哪里移动”。
- 如果没有它们，归一化后的表示会被固定在过于僵硬的分布上。

### 补充
从表达能力看，`γ` 和 `β` 可以理解为“把标准化后的表示重新调回任务需要的空间”。

---

## 4. 为什么要用 RMSNorm？相比 LayerNorm 有什么优势？

### 问题
为什么很多 LLM 从 LayerNorm 转向 RMSNorm？

### 核心结论
RMSNorm 去掉了均值中心化，只保留基于均方根的尺度归一化。这样计算更简单，通常也足以提供训练稳定性，因此很适合大模型。

### 公式

#### LayerNorm
$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

#### RMSNorm
设输入维度为 `d`，则：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}
$$

有些实现不使用 `\beta`。

### 解释
- **少一次中心化操作**：不需要计算并减去均值。
- **更省计算**：在大规模训练时更有价值。
- **保留主要稳定性收益**：很多场景中，控制尺度比强制中心化更关键。
- **对缩放更稳定**：若输入整体被放大，归一化后输出仍能维持稳定尺度。

### 补充
RMSNorm 不是在所有场景都绝对优于 LayerNorm，但在现代 LLM 中，它常常是更划算的工程选择。
