# 2024第四季度 论文学习记录

## 去雾+Mamba/visionRWKV

论文笔记内容：
① 解决什么问题？
② 创新点和方法是什么？
③ 结果和结论是什么？

泛读：
① 看论文摘要
② 看论文创新点（方法和图）
③ 看实验的表 看性能

精读：
看introduction relate-work experiment都看 做笔记
paper哪个地方最work的，怎么发现这个方法的

做实验：看相关方向顶会的一些baseline

## 1 Survey

### 1.1 Mamba

#### 1.1.1 [A Survey of Mamba](https://arxiv.org/abs/2408.01129)

##### 1.1.1.1 挑战

**Transformer**
① Transformer仍然面临着固有的局限性,尤其是注意力计算的二次计算复杂性导致推断过程非常耗时。
② 当涉及到从基于Transformer的模型生成响应或预测时,推理过程可能会很耗时。例如,语言模型的自动回归设计需要按顺序生成输出序列中的每个标记,这需要在每个步骤重复计算注意力得分,从而导致推理时间变慢。

**RNN**
① RNN在有效提取输入序列中的长程动态方面能力有限。当信息在连续的时间步中传递时,网络中权重的重复乘法会导致信息稀释或丢失。
② RNN以增量方式处理序列数据,这限制了它们的计算效率,因为每个时间步都依赖于前一个时间步。这使得并行计算对它们来说具有挑战性。
③ 传统的RNN缺乏内置的注意力机制,阻碍了网络对数据的关键部分进行选择性建模的能力。
④ 传统的RNN在非线性循环框架内运行,其中每个计算仅取决于先前的隐藏状态和当前输入。虽然这种格式允许RNN在自动回归推理期间快速生成输出,但它阻碍了它们充分利用GPU并行计算的能力,导致模型训练速度变慢。

**GNN**
这些模型面临一个被称为过度平滑的重大局限性(Chen等人,2020),尤其是在试图捕获高阶邻接信号时。

**CNN**
基于CNN的方法受局部感受野的限制,导致在捕捉全局和长距离语义时表现欠佳(Gu and Dao, 2023)。

**SSMs**
最传统的SSM是时间不变的,这意味着它们的𝐀、𝐁、𝐂和Δ与模型输入𝑥无关。这将限制上下文感知建模,从而导致SSM在某些任务中的性能较差,例如选择性复制。

**Mamba**
曼巴架构仍然面临一些挑战,例如内存损耗、对不同任务的泛化以及与基于Transformer的语言模型相比,捕捉复杂模式的能力较差。

##### 1.1.1.2 主要贡献

① 基于Mamba的模型的进展
② 使Mamba适应不同数据的技术
③ Mamba可以发挥优势的应用

具体来说,我们首先回顾了各种代表性深度学习模型的基础知识以及Mamba-1和Mamba-2的细节作为前期工作。然后,为了展示Mamba对人工智能的重要性,我们全面回顾了相关研究,重点关注Mamba模型的架构设计、数据适应性和应用。最后,我们讨论了当前的局限性,并探讨了各种有前途的研究方向,为未来的研究提供更深入的见解。

##### 1.1.1.3 知识点

**RNN**
① RNN是非线性循环模型,通过利用隐藏状态中存储的历史知识来有效地捕捉时间模式。
② 基于RNN的模型在捕捉时间动态方面取得了卓越的成果。
  
**GNN**
① GNN在通过消息传递机制捕获相邻关系方面展现出巨大的潜力,其中信息通过堆叠层在连接图上传播。

**Transformer**
① 自然语言处理中,Self-Attention使模型能够理解序列中不位置之间的关系。
②  In multi-head attention,输入序列由多组自我注意力模块并行处理。每个模块独立运行,执行与标准自我注意力机制完全相同的计算。然后,将每个模块的注意力权重相加,得到一个值向量的加权总和。通过这一聚合步骤,模型能够利用来自多个模块的信息,捕捉输入序列中的不同模式和关系。

**SSMs**
① 与RNN不同,SSMs是线性模型,具有关联性。
② 与仅能支持一种计算的RNN和Transformer不同,离散SSM具有线性特性,因此能够灵活地支持循环和卷积计算。

**Mamba**
*引入了三种基于结构化状态空间模型(Structured State Space Models)的创新技术*
  ①初始化策略(HIPPO)建立了一个连贯的隐藏状态矩阵,有效地促进了长程记忆。
  ②选择机制使SSM能够获得内容感知表示。
  ③Mamba设计了两种硬件感知计算算法,即并行关联扫描和内存重计算,以提高训练效率。

Mamba主要关注隐藏状态矩阵𝐀的初始化,以捕获复杂的时间依赖性。这是通过利用HiPPO理论(Gu et al., 2020)和创新的缩放勒让德测量(LegS)实现的,确保仔细考虑完整的历史背景,而不是有限的滑动窗口。
它在不同输入时间尺度下保持一致,并且计算速度很快 (Gu et al., 2020)。此外,它还具有有界梯度和近似误差,有助于参数学习过程。

Mamba 中时变选择机制的结构与 Transformer 中的注意力机制类似,即两者都根据输入及其投影执行操作,这使得 Mamba 的 SSM 能够实现灵活的内容感知建模。然而,它失去了与卷积的等效性,这对它的效率产生了负面影响。

并行关联扫描将模型训练的计算复杂度从$𝐎(𝑁^2𝑑)$降低到$𝐎(𝑁/𝑡)$。从本质上讲,扫描的核心是在给定的输入上构建一个平衡的二叉树,并从根开始遍历。换句话说,并行关联扫描首先从叶子到根遍历(即向上扫描),在树的内部节点创建部分和。然后,它反转遍历,从根向上移动,使用部分和构建整个扫描(即向下扫描)。
另一方面,Mamba利用传统的重计算方法来减少训练选择性SSM层所需的总体内存.

##### 1.1.1.4 Mamba改进

① 块设计：整合、替换、修改
② 扫描模式:
平面扫描-Bidirectional Scan, Sweeping Scan, Continuous Scan, and Efficient Scan.
立体扫描-Hierarchical Scan, Spatiotemporal Scan, and Hybrid Scan.
③ 内存管理：存储器初始化、压缩和连接。

##### 1.1.1.5 机遇

Mamba具有Transformer的内容感知学习功能,同时能够根据输入长度线性扩展计算量,从而有效地捕捉长距离依赖关系,提高训练和推理的效率。

通过连接SSM和注意力,Mamba-2引入的SSD框架(Dao和Gu,2024)允许我们为Transformer和Mamba开发共享的词汇和技术库。

#### 1.1.2 [A Survey on Visual Mamba](https://arxiv.org/abs/2404.15956)  CVPR

##### 1.1.2.1 主要贡献

① 这篇调查论文首次对视觉领域的Mamba技术进行了全面回顾，并明确侧重于分析提出的策略。
② 在基于Naive的Mamba视觉框架的基础上，我们研究了如何增强Mamba的功能并将其与其他架构相结合以实现卓越的性能。
③ 我们根据各种应用任务对文献进行了整理，从而进行了深入的探索。我们建立了分类法，确定了每种任务特有的进展，并提供了克服挑战的见解。

##### 1.1.2.2 Mamba优势

RNN和LSTM在处理渐进梯度和长程依赖性时存在困难，而Mamba则能提供高效的计算和内存利用率。

##### 1.1.2.3 Mamba相关

① Pure mamba:
 [Vision Mamba](https://arxiv.org/abs/2401.09417)-Vim-based, [LocalMamba](https://arxiv.org/abs/2403.09338)-Vim-based, [VMamba](https://arxiv.org/abs/2401.10166)-VSS-based, [PlainMamba](https://arxiv.org/abs/2403.17695)-VSS-based.

##### 1.1.2.4 扫描方向总结

BiDirectional Scan 双向扫描 [26]
Cross-Scan 交叉扫描[27]
Continuous 2D Scanning 连续2D扫描 [29]
Local Scan 局部扫描 [28]
Efficient 2D Scanning (ES2D) 高效二维扫描 [30]
Omnidirectional Selective Scan 全向选择性扫描 [35]
3D BiDirectional Scan 3D双向扫描 [36]
Hierarchical Scan 分层扫描 [37]
Spatiotemporal Selective Scan 时空选择性扫描 [38]
Multi-Path Scan 多路径扫描 [39]

#### 1.1.3 [Visual Mamba: A Survey and New Outlooks](https://arxiv.org/abs/2404.18861)

CNN:局部感受野，限制空间上下文捕捉
ViT:补丁二次计算成本高
Mamba：线性计算成本、类似于Transformer的建模能力

##### 1.1.3.1 公式

连续化 经典状态空间模型
  $$\begin{equation}h'(t) = Ah(t)+Bx(t), y(t) = Ch(t)     \end{equation}      $$
在将$𝐀,𝐁$离散化为$\bar𝐀$,$\bar𝐁$ 之后，方程（1）可以重新表述为
   $$\begin{equation}h_t = \bar Ah_{t-1}+\bar     Bx_t, y_t = Ch_t     \end{equation}      $$
重新表述，计算为卷积
$$ \begin{equation}
  \bar K = (\bar C \bar B, C\bar A \bar B,...,\bar C \bar A^{L-1} \bar B), y=x* \bar K
\end{equation}$$
𝐿表示输入序列𝒙的长度， ∗代表卷积操作。向量 $\bar𝐊$∈ℝ。 𝐿是SSM 卷积核，能够实现序列输出的同时合成。给定 $\bar𝐊$，方程（3）中的卷积操作可以使用快速傅里叶变换（FFT）高效计算。

SSM 中的参数由公式（1）、公式（2）、公式（3）指示，与输入或时间动态无关。

##### 1.1.3.2 扫描
扫描方向
扫描轴线
扫描连续性
扫描采样
①扫描方向（单向、双向...）：解决视觉序列的非因果特性。
②扫描轴线（水平、垂直、左对角、右对角、深度（3维）、超越空间轴（沿着通道））：处理视觉数据固有的高维性。
③扫描连续性（之字形、希尔伯特扫描、重新排序扫描...）：考虑扫描路径上补丁的空间连续性。
④扫描采样（本地采样、多尺度采样...）：将完整图像划分为子图像以捕捉空间信息。
##### 1.1.3.3 Tokenization
二维图像通过茎模块转换为视觉标记序列。向视觉标记添加位置嵌入（optional）
一维or二维

##### 1.1.3.4 应用
有图像重建

##### 1.1.3.5未来展望

Mamba架构在扩展到大型网络配置时面临稳定性问题
自回归预训练策略处于萌芽阶段
Mamba 可以选择性地忽略不相关的信息，随着上下文长度的增加而产生一致的性能改进。
Mamba 架构的效率和功效为数据可扩展性带来了巨大的机会。
Mamba 模型在不依赖大规模数据集的情况下具有实现最佳性能的巨大潜力。这为开发小型但有效的基于 Mamba 的模型提供了机会。
####  Table 3：Comparison of different backbones on ImageNet-1K [23] classification

### 1.2 dehazing

## 2 conference paper

### 2.1 Mamba/visionRWKV

#### 2.1.1 [Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures](https://arxiv.org/abs/2403.02308)(OpenGVLab)
与 ViT 相关的二次计算复杂性限制了它们有效处理高分辨率图像和冗长序列的能力

##### 2.1.1.1 主要贡献
(1)我们提出VRWKV作为ViT的低成本替代方案，以更低的计算成本实现全面替代。我们的模型不仅保留了 ViT 的优点，包括捕获远程依赖关系的能力和处理稀疏输入的灵活性，而且还将复杂性降低到线性水平。

(2)为了适应视觉任务，我们引入了双向全局注意力和一种称为 Q-Shift 的新颖令牌移位方法，从而实现了全局注意力的线性复杂性。

(3)我们的模型超越了基于窗口的 ViT，并且与全局注意力 ViT 相当，随着分辨率的提高，展示了更低的 FLOP 和更快的处理速度。

##### 2.1.1.2 相关工作
窗口注意力[ 30,51,6 ]将自注意力计算限制在局部窗口内，大大降低了计算复杂度，同时通过窗口级交互保留了感受野。

##### 2.1.1.3 整体架构
在原来的RWKV基础上修改
① 原来的因果注意力转变为双向全局注意力。
② 相对偏差：我们计算时间差的绝对值 𝑡−𝑖 并将其除以token总数（表示为𝑇）来表示不同尺寸图像中标记的相对偏差。
③ 灵活的衰减：我们不再限制可学习的衰减参数𝑤在指数项中为正，允许指数衰减注意力集中在tokens在不同通道中远离当前的token上。

Q-Shift
然而，一维衰减与二维图像中的相邻关系并不相符。因此，我们在每个空间混合和通道混合模块的第一步中引入了四向令牌移位（Q-Shift）。
Q-Shift 操作允许所有标记与其相邻标记进行移位和线性插值

有界指数：随着输入分辨率的增加，指数衰减和增长都很快超出浮点数的范围。因此，我们将指数项除以token数量，使最大衰减和增长受到限制。
当模型变深时，我们在注意力机制和Squared ReLU操作之后直接添加层归一化[ 2 ]，以防止模型的输出溢出。
这两项修改可以稳定地缩放输入分辨率和模型深度，从而允许大型模型稳定地训练和收敛。
#### 2.1.2 [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)(ICML2024 accept)
早期纯Mamba backbone
在本文中，我们证明了视觉表示学习中对自注意力的依赖是不必要的，并提出了一种具有双向 Mamba 块的新通用视觉主干 (Vim)。
它用位置嵌入来标记图像序列，并用双向状态空间模型压缩视觉表示。
我们构建了一个基于纯 SSM 的模型，可以将其用作通用视觉主干。
相比VMamba，Vim 主要专注于视觉序列学习，并拥有多模态数据的统一表示。
##### 2.1.2.1 主要贡献
① 我们提出Vision Mamba (Vim)，它结合了用于数据依赖的全局视觉上下文建模的双向SSM 和用于位置感知视觉理解的位置嵌入。
② 无需注意力机制，所提出的Vim 具有与ViT 相同的建模能力，但仅具有次二次方时间计算和线性内存复杂度。
具体来说，Vim在执行批量推理以提取 1248x1248 分辨率图像特征时比DeiT快2.8倍，并节省 86.8% GPU 内存。
③ 我们对ImageNet 分类和密集预测下游任务进行了广泛的实验。结果表明，与成熟且高度优化的普通视觉 Transformer（即 DeiT）相比，Vim 实现了卓越的性能。
##### 2.1.2.2 实验
① 图像分类
ImageNet-1K 
  Vim-Small top-1准确率 80.3 比 ResNet50 高 4.1 个百分点。
  多项好于DeiT

##### 2.1.2.3 总结
Vim 以序列建模方式学习视觉表示，并且不会引入特定于图像的归纳偏差。得益于所提出的双向状态空间建模，Vim 实现了数据依赖的全局视觉上下文，并享有与 Transformer 相同的建模能力，同时具有较低的计算复杂度。受益于 Mamba 的硬件感知设计，Vim 在处理高分辨率图像时，推理速度和内存占用明显优于 ViT。标准计算机视觉基准上的实验结果验证了 Vim 的建模能力和高效率，表明 Vim 具有成为下一代视觉骨干的巨大潜力。
#### 2.1.3 [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)(NeurIPS2024 spotlight)
早期纯Mamba
通过将 Mamba 与多向扫描和分层网络架构相结合，在视觉识别方面展示了令人印象深刻的结果。
#### 2.1.4 [Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model](https://arxiv.org/abs/2405.14174)(NeurIPS 2024)
每个模块中引入卷积前馈网络，以增强通道间信息交换和局部特征提取。

#### 2.1.5[U-shaped Vision Mamba for Single Image Dehazing](https://arxiv.org/abs/2402.04139)
Mamba去雾

#### 2.1.5[RSDehamba: Lightweight Vision Mamba for Remote Sensing Satellite Image Dehazing](https://arxiv.org/abs/2405.10030)
Mamba去雾

#### 2.1.6[MambaIR: A Simple Baseline for Image Restoration with State-Space Model](https://arxiv.org/abs/2402.15648)(ECCV2024)
Mamba超分辨率
#### 2.1.7[Activating Wider Areas in Image Super-Resolution](https://arxiv.org/abs/2403.08330)
### 2.2 dehazing
