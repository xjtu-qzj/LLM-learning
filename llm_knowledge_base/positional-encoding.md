# Positional Encoding

## 1. 为什么“固定距离 k 对应固定线性变换”很重要？

### 问题
为什么会强调：对于任意位置 `pos`，存在一个与 `pos` 无关、只与偏移 `k` 有关的线性变换，使得 `PE(pos)` 能映射到 `PE(pos + k)`？

### 核心结论
这类性质说明位置编码天然支持相对位置信息建模。模型不只知道“你在第几个位置”，还更容易知道“你和另一个 token 相距多远”。

### 公式
对于某种位置编码 `PE`，若存在线性变换 `A_k`，使得：

$$
PE(pos + k) = A_k \, PE(pos)
$$

则说明固定偏移 `k` 可以由统一变换表示。

对于经典正弦位置编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

利用三角恒等式：

$$
\sin(a+b) = \sin a \cos b + \cos a \sin b
$$

$$
\cos(a+b) = \cos a \cos b - \sin a \sin b
$$

可以把 `pos + k` 的编码表示为 `pos` 编码经过一个旋转矩阵后的结果。

### 解释
- 这说明正弦位置编码不是简单的“位置标签表”。
- 它的结构允许模型通过线性方式感知相对位移。
- 因此，注意力层更容易学习“前一个词”“后两个词”这类关系。

### 补充
这个性质也是后续理解 RoPE 的关键前置概念。

---

## 2. 正弦位置编码为什么适合表示相对位置？

### 问题
为什么 Transformer 最初使用的正弦位置编码，能在一定程度上表达相对位置信息？

### 核心结论
因为它由不同频率的正弦和余弦组成，而正弦余弦天然满足平移到旋转的关系，所以相对偏移可以被编码成规则的线性变换。

### 公式
对第 `i` 个频率，记：

$$
\theta_i(pos) = \frac{pos}{10000^{2i/d}}
$$

则每一对维度可写成：

$$
\begin{bmatrix}
\sin \theta_i(pos) \\
\cos \theta_i(pos)
\end{bmatrix}
$$

位置从 `pos` 变到 `pos + k` 时，相当于角度增加 `\theta_i(k)`，因此：

$$
\begin{bmatrix}
\sin \theta_i(pos+k) \\
\cos \theta_i(pos+k)
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta_i(k) & \sin \theta_i(k) \\
-\sin \theta_i(k) & \cos \theta_i(k)
\end{bmatrix}
\begin{bmatrix}
\sin \theta_i(pos) \\
\cos \theta_i(pos)
\end{bmatrix}
$$

### 解释
- 每个频率子空间都像一个二维旋转平面。
- 相对位移对应旋转角变化。
- 多个频率叠加后，模型能同时感知短程和长程位置关系。

---

## 3. RoPE 的核心思想是什么？

### 问题
RoPE 和传统“加法位置编码”有什么本质区别？

### 核心结论
RoPE 不把位置向量直接加到 token embedding 上，而是对 Query 和 Key 的向量对进行旋转，使位置关系直接进入注意力点积结构。

### 公式
设二维向量对为：

$$
x = \begin{bmatrix}x_{2i} \\ x_{2i+1}\end{bmatrix}
$$

位置 `p` 的旋转矩阵为：

$$
R_{\theta(p)} =
\begin{bmatrix}
\cos \theta(p) & -\sin \theta(p) \\
\sin \theta(p) & \cos \theta(p)
\end{bmatrix}
$$

则旋转后：

$$
\text{RoPE}(x, p) = R_{\theta(p)} x
$$

应用到注意力时，可写成：

$$
q'_p = R_{\theta(p)} q_p, \quad k'_t = R_{\theta(t)} k_t
$$

注意力分数：

$$
{q'_p}^{\top} k'_t
$$

这个结果会自然依赖于相对位置 `p - t`。

### 解释
- RoPE 不是把位置编码作为附加信息拼上去，而是直接改写向量几何结构。
- Query 和 Key 同时旋转后，内积中会保留相对位移信息。
- 这使 RoPE 在长序列建模中通常比绝对位置编码更自然。

### 补充
很多现代 LLM 采用 RoPE，也是因为它能较好兼容自回归注意力，并在长上下文下表现稳定。
