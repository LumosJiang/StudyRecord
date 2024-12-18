# 论文学习记录

## 1 去雾

### 1.1 夜间去雾

#### 1.1.1 [A Semi-supervised Nighttime Dehazing Baseline with Spatial-Frequency Aware and Realistic Brightness Constraint](https://arxiv.org/abs/2403.18548)(CVPR2024)

##### 提出了一个空间和频域感知的半监督夜间除雾网络（SFSNiD）可以消除伴随着辉光和噪音的夜间雾霾

###### 设计了一个空间和频域信息交互（SFII）模块来同时处理具有局部、耦合和频率不一致特性的雾霾、辉光和噪声。多通道幅度和相位谱被动态过滤和聚合。空间和频域特征通过局部注意力集成

① 采用局部注意力来学习空间域中的归纳偏差以抑制局部失真。频谱动态滤波策略旨在处理频率特性不一致的失真。考虑到这些失真的耦合，空间和频率信息被集成为双域交互模块，用于特征提取和图像重建。 ② 旨在抑制失真，同时实现真实的亮度。游戏引擎提供的模拟数据用于生成伪标签，可以抑制再训练过程中的雾霾和辉光。 ③ 采用真实世界的模糊图像作为真实亮度约束的真实亮度信号。

#### 1.1.2 [NightHazeFormer: Single Nighttime Haze Removal Using Prior Query Transformer](https://arxiv.org/abs/2305.09533)(CVPR2023)

##### 基于端到端的transformer框架 在现实世界的夜间雾霾成像条件下，照明以霓虹灯等各种人造光源为主，它们具有不同的位置和颜色，亮度范围有限。因此，获取的降质图像除了雾霾外，还会受到多重散射、光照不均、辉光、模糊、隐藏噪声等影响

###### 我们在 Transformer 解码器中引入了两个强大的物理先验：暗通道先验 (DCP) （He 等人， 2011 ）和亮通道先验（BCP） （Wang 等人， 2013 ） ，以生成非-可学习的先前查询 为了进行微调，我们采用合成域中的预训练模型以无监督的方式生成粗糙的无雾图像。然后，采用一种称为BCCR （Meng et al . , 2013 ）的有效去雾方法对它们进行去雾以提高可见度。最后，将获得的伪地面实况与现实世界的夜间模糊图像相结合，并在合成域中进行微调，以减少合成域与真实域之间的差异

① 我们提出了一种基于端到端Transformer的网络，称为 NightHazeFormer，用于夜间除雾。通过将两个强大的先验合并到 Transformer 解码器中，我们的 NightHazeFormer 生成不可学习的先验查询，有效指导我们的网络从输入的夜间模糊图像中学习丰富的先验特征。 ② 开发了半监督微调训练范式以提高泛化能力。我们将现实世界的夜间模糊图像与生成的伪地面真实标签相结合，然后将其输入合成域以微调预训练模型并使其能够学习真实数据的领域知识。 ③ 为了弥补现有数据集中退化模拟的不足，我们提出了一个名为 UNREAL-NH 的大规模合成夜间模糊图像数据集。我们的 UNREAL-NH 考虑了多种类型的退化并解决了现有数据集的局限性。

#### 1.1.3 [Enhancing Visibility in Nighttime Haze Images Using Guided APSF and Gradient Adaptive Convolution](https://arxiv.org/html/2308.01738v4)(ACM MM2023)

##### 现有的夜间除雾方法通常难以处理辉光或低光条件，导致视觉效果过暗或辉光输出不受抑制

###### 提出了一个光源感知网络来检测夜间图像的光源，然后是 APSF（大气点扩散函数）引导的发光渲染。然后我们的框架在渲染图像上进行训练，从而产生辉光抑制。此外，我们利用梯度自适应卷积来捕获模糊场景中的边缘和纹理。通过利用提取的边缘和纹理，我们增强了场景的对比度，而不会丢失重要的结构细节。为了提高低光强度，我们的网络学习注意力图，然后通过伽玛校正进行调整。此注意力对低光区域具有高值，对雾霾和辉光区域具有低值

① 据我们所知，我们的方法是第一个基于学习的网络，可以一次性处理夜光和弱光条件。 ② 我们提出了一个光源感知网络和 APSF 引导的发光渲染来模拟来自不同光源的发光效果。通过学习 APSF 引导的辉光渲染数据，我们的框架有效地抑制了现实世界模糊图像中的辉光效果。 ③ 由于夜间图像对比度较低，我们采用梯度自适应卷积进行边缘增强，并采用双边内核进行纹理增强。

### 1.2 日间去雾

大气散射模型 I(x)\=J(x)T(x)+A(x)(1−T(x))I(x)=J(x)T(x)+A(x)(1-T(x))I(x)\=J(x)T(x)+A(x)(1−T(x)) 其中I(x)为带雾图像，J(x)为清晰图像，T(x)为透射率，A为全局全局背景光。

通常定义T(x)\=e−βd(x)T(x)=e^{-\\beta d(x)}T(x)\=e−βd(x)

其中β\\betaβ为大气散射系数,d(x)d(x)d(x)为相机到物体深度。

#### 1.2.1 [Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing](https://arxiv.org/abs/2403.01105)(CVPR2024)

##### 通过双任务交互机制将深度估计和去雾集成在一起，实现了两者性能的相互增强

###### 端到端图像去雾方法直接从有雾图像中恢复清晰图像，无需借助大气散射模型缺乏大气散射模型的指导也给恢复清晰图像带来了挑战。为了解决这个问题，提出了基于先验信息的图像去雾

① 该方法通过感知去雾网络的输出结果与预期结果之间的差异来改进有雾图像的深度估计，使得去雾网络能够接收高质量的深度估计信息作为去雾过程的指导。 ② 在提高去雾性能的深度估计方面，它通过感知去雾结果与理想图像之间的深度信息差异，使去雾网络关注去雾效果可能不理想的区域。 ③ 在去雾网络促进深度估计方面，通过使深度估计网络关注非理想去雾区域，在有雾图像上获得更准确的预测结果，提高了深度估计的鲁棒性。

#### 1.2.2 [ODCR: Orthogonal Decoupling Contrastive Regularization for Unpaired Image Dehazing](https://arxiv.org/abs/2404.17825v1)(CVPR2024)

##### 我们的方法基于这样的假设：图像由影响雾霾程度的与雾霾相关的特征以及与雾霾无关的特征（例如纹理和语义信息）组成

###### ODCR 旨在确保去雾结果中与雾霾相关的特征与清晰图像的特征非常相似，而与雾霾无关的特征与输入雾霾图像一致

###### 获取具有相同背景的雾/清晰图像对具有挑战 本文针对不成对图像去雾（UID）

① 提出了在 Stiefel 流形上进行几何优化的正交 MLP，它可以将图像特征投影到正交空间中，从而减少不同特征之间的相关性 ② 此外，提出了一种任务驱动的深度特征分类器（DWFC），它根据每个通道特征在以自监督方式预测特征源是模糊还是清晰时的贡献为正交特征分配权重。 ③ 最后，引入加权 PatchNCE (WPNCE) 损失，以实现将输出图像中与雾霾相关的特征拉向清晰图像的特征，同时使与雾霾无关的特征接近雾化输入的特征。

#### 1.2.3 [Contrastive Learning for Compact Single Image Dehazing](https://arxiv.org/abs/2104.09367)(CVPR2021)

##### 我们提出了一种基于对比学习的新型对比正则化（CR constrastive regulation），以分别利用模糊图像和清晰图像的信息作为负样本和正样本。 CR确保恢复的图像在表示空间中被拉近清晰图像并远离模糊图像

###### 仅正向去雾目标函数效果较差。大多数现有方法通常采用清晰图像（又名ground-truth ）作为正样本1 通过基于 L1/L2 的图像重建损失来指导去雾网络的训练，无需任何正则化。然而，仅图像重建损失无法有效处理图像的细节，这可能导致恢复图像的颜色失真

###### 参数较多的去雾网络。之前的工作专注于通过显着增加去雾模型的深度或宽度来提高去雾性能，而不考虑内存或计算开销，这阻碍了它们在资源有限的环境（例如移动设备）上的使用或嵌入式设备

① 我们提出了一种新颖的 ACER-Net，通过对比正则化和基于高度紧凑的类自动编码器的去雾网络来有效生成高质量的无雾图像。与最先进的方法相比，AECR-Net 实现了最佳的参数性能权衡。 ② 所提出的对比正则化作为通用正则化可以进一步提高各种最先进的去雾网络的性能。 ③ 所提出的类自动编码器（AE）去雾网络中的自适应混合和动态特征增强模块可以分别帮助去雾模型自适应地保留信息流并增强网络的变换能力。

#### 1.2.4 [Single UHD Image Dehazing via Interpretable Pyramid Network](https://arxiv.org/abs/2202.08589)(CVPR2022)

##### 目前，大多数单图像去雾模型无法使用单个 GPU 着色器实时运行超高分辨率 (UHD) 图像。为了解决这个问题，我们引入了拉普拉斯金字塔模式无限逼近泰勒定理的原理，建立了一个能够实时处理4K模糊图像的模型

###### 基于物理先验的去雾算法需要很长的时间，并且在运行高分辨率有雾图像时性能较差。例如DCP、non-local都需要超过100s来处理单个4K图像

###### 基于深度学习的方法以计算资源为代价采用多个卷积运算来获得模型的高性能不幸的是，它需要超过 0.1 秒，并且无法使用 24G RAM 上的全分辨率图像进行大规模训练

① 我们用拉普拉斯金字塔结构引入了泰勒定理的无限逼近。这种方法不仅具有极高的性能（实时运行 4K 图像），而且具有很强的可解释性。

② 我们提出了一种基于塔克重建的正则化项，进一步限制了特征空间中异常信号的生成。

③ 我们提出注意力共享张量𝐾它不仅关注真实的纹理细节，还减少了高分辨率图像的计算时间。

#### 1.2.5 [Vision Transformers for Single Image Dehazing](https://arxiv.org/abs/2204.03883)(CVPR2022)

##### 我们发现视觉 Transformer 中常用的 LayerNorm和GELU会损害图像去雾性能。具体来说，视觉 Transformer 中使用的 LayerNorm 分别对图像块对应的 token 进行归一化，导致块之间的相关性丢失

① 我们建议使用RescaleNorm和ReLU来替代常用的LayerNorm和GELU，以避免一些对高级视觉任务不重要但对低级视觉任务至关重要的负面影响。 ② 为了提高 MHSA 的能力，我们提出了一种基于反射填充的移位窗口划分方案和一种使用卷积与注意力并行的空间信息聚合方案。 ③ 我们收集了大规模遥感图像去雾数据集来评估网络去除高度非均匀雾霾的能力

#### 1.2.6 [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)(CVPR2021)

##### 在现有的基于 Transformer 的模型中，token都是固定规模的，这种属性不适合这些视觉应用。另一个区别是图像中像素的分辨率比文本段落中的单词要高得多。存在许多视觉任务，例如语义分割，需要在像素级进行密集预测，这对于高分辨率图像上的 Transformer 来说是很棘手的，因为其自注意力的计算复杂度与图像大小成二次方

###### 我们提出了一个通用的 Transformer 主干，称为 Swin Transformer，它构建分层特征图，并且具有与图像大小线性的计算复杂度

Swin Transformer 通过从小尺寸补丁（灰色轮廓）开始并逐渐合并更深 Transformer 层中的相邻补丁来构建分层表示。 借助这些分层特征图，Swin Transformer 模型可以方便地利用先进技术进行密集预测，例如特征金字塔网络（FPN） \[ 42 \]或U-Net \[ 51 \] 。线性计算复杂度是通过在分割图像的非重叠窗口（以红色框出）内本地计算自注意力来实现的。每个窗口中的补丁数量是固定的，因此复杂度与图像大小成线性关系。这些优点使 Swin Transformer 适合作为各种视觉任务的通用骨干网，与之前基于 Transformer 的架构\[ 20 \]形成鲜明对比，后者生成单一分辨率的特征图并具有二次复杂度。

#### 1.2.7 [MB-TaylorFormer: Multi-branch Efficient Transformer Expanded by Taylor Formula for Image Dehazing](https://arxiv.org/abs/2308.14036)(ICCV2023)

##### 在图像去雾领域，直接应用Transformer存在几个挑战：1）Transformer的计算复杂度与特征图的分辨率成二次方，这使得它不太适合像素到像素的去雾任务。尽管有些工作在小空间窗口中应用自注意力\[ 35 , 73 \]来缓解这个问题，但 Transformer 的感受野受到限制； 2）视觉Transformer的基本元素通常具有更灵活的尺度\[ 39 \] 。然而，现有的视觉 Transformer 网络\[ 78 , 73 \]通常通过固定的卷积核生成固定规模的标记。因此，通过将灵活的补丁嵌入 Transformer 引入去雾任务中，仍有改进的空间

###### 我们的模型被称为泰勒公式扩展的多分支变换器（MB-TaylorFormer），可以在补丁嵌入阶段更灵活地嵌入粗到细的特征，并以有限的计算成本捕获长距离像素交互

① 我们提出了一种基于泰勒展开的线性化 Transformer 的新变体，以模拟像素之间的长距离交互，而无需窗口分割。引入MSAR模块来进一步纠正TaylorFormer的self-attention中的错误。 ② 我们设计了具有多尺度补丁嵌入的多分支架构。其中，多种字段大小、灵活的感受野形状以及多层次的语义信息可以帮助同时生成多尺度的token并捕获更强大的特征。 ③ 在公共合成和真实去雾数据集上的实验结果表明，所提出的MB-TaylorForme 以很少的参数和 MAC 实现了最先进的 (SOTA) 性能。

## 2 去雨

### 2.1 Transformer

#### 2.1.1 [Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining](https://arxiv.org/abs/2404.01547)

##### 我们开发了一种端到端的多尺度 Transformer，它利用各种尺度的潜在有用特征来促进高质量的图像重建

① 我们设计了一个有效的多尺度Transformer，通过开发和利用多尺度降雨信息来生成高质量的除雨结果。 ② 我们引入隐式神经表示来更好地学习常见的降雨退化特征，并表明它可以帮助促进除雨并增强复杂场景中除雨模型的鲁棒性。 ③ 我们将简单而有效的双向反馈传播操作集成到我们的多尺度 Transformer 中，以实现更好的跨尺度特征交互。

#### 2.1.2[Learning A Sparse Transformer Network for Effective Image Deraining](https://arxiv.org/abs/2303.11950)

① 我们提出了一种稀疏 Transformer 架构，以帮助生成高质量的去雨结果，并具有更准确的细节和纹理恢复。 ② 我们开发了一个简单而有效的可学习的 top- k选择算子，以自适应地维护最有用的自注意力值，以实现更好的特征聚合。 ③ 我们设计了一种基于混合尺度融合策略的有效前馈网络，以探索多尺度表示，以更好地促进图像去雨。

## 3 去模糊

### 3.1 Transformer

#### 3.1.1 [A Unified Framework for Microscopy Defocus Deblur with Multi-Pyramid Transformer and Contrastive Learning](https://arxiv.org/abs/2403.02611)

##### 显微镜散焦去模糊

###### ① 由于显微镜图像和自然场景图像的特征之间存在显着差异，显微镜去模糊提出了与现实世界去模糊任务不同的挑战

###### ② 显微镜去模糊的另一个问题是训练鲁棒模型的数据不足

1）首次提出多金字塔变压器（ MPT ）用于显微镜散焦去模糊。它使用提出的跨尺度窗口注意力（ CSWA ）和二次放大的感受野来对每个显式金字塔中的局部尺度和缩小尺度图之间的远程空间注意力进行建模，以适应显微镜数据集的较长注意力跨度。

2）提出了尺度内通道注意力（ ISCA ），通过所提出的特征增强前馈网络（ FEFN ）将全局通道上下文合并到CSWA空间信息中，为金字塔提供额外的尺度内通道特征。

3）提出了一种具有扩展频率对比正则化（ EFCR ）的训练策略，通过合成再模糊利用超出像素约束的潜在去模糊信号来缓解数据不足，这是显微镜去模糊中对比学习的首次实现。它还支持跨域去模糊知识传输，促进额外的数据训练并增强未标记图像去模糊。

### 3.2 Others

#### 3.2.1 [Blur2Blur: Blur Conversion for Unsupervised Image Deblurring on Unknown Domains](https://arxiv.org/abs/2403.16205)

##### 本文提出了一种创新框架，旨在训练适合特定相机设备的图像去模糊算法。该算法的工作原理是将难以去模糊的模糊输入图像转换为另一幅更适合去模糊的模糊图像

###### 使用不成对的数据进行无监督学习。这些方法很难有效地弥合这些领域之间的差距，因为（1）不同图像之间的模糊程度存在显着差异，这会影响内部对象的感知语义，以及（2）图像的复杂性和不可预测性。现实世界的模糊模式，通常与这些模型中使用的简单假设相矛盾。因此，实现真正的盲图像去模糊的挑战仍未解决

考虑到这些限制，我们的 Blur2Blur 方法以模糊内核传输的创新理念为中心。这涉及将任何特定相机的模糊内核转换为具有强大的预训练去模糊模型的数据集或相机的熟悉模糊内核。这种方法使我们能够在无监督框架内利用监督技术的优势，有效解决具有各种未知模糊分布的去模糊图像的挑战。

#### 3.2.2 [ID-Blau: Image Deblurring by Implicit Diffusion-based reBLurring AUgmentation](https://arxiv.org/abs/2312.10998)

##### 本文提出了基于隐式扩散的重新模糊增强（ID-Blau），利用清晰图像与可控模糊条件图配对来生成相应的模糊图像。我们将模糊图像的模糊模式及其方向和幅度参数化为像素级模糊条件图，以模拟运动轨迹并在连续空间中隐式表示它们。通过对不同的模糊条件进行采样，ID-Blau 可以生成训练集中未见过的各种模糊图像

① 我们提出ID-Blau，一种稳定可控的模糊增强策略，用于增强动态场景图像去模糊。 ② 我们对连续模糊条件场进行建模，以隐式表示模糊方向和幅度，其中我们可以对各种像素级模糊条件图进行采样，以生成训练集中未提供的各种重新模糊图像。 ③ 所提出的ID-Blau 将像素级模糊条件图集成到扩散模型中，以生成高质量的再模糊图像。

#### 3.2.3 [LDP: Language-driven Dual-Pixel Image Defocus Deblurring Network](https://arxiv.org/abs/2307.09815)

##### 从单个图像中进行散焦去模糊是一个不适定问题。与直接恢复全焦点图像的方法相比，基于模糊图的去模糊方法已经证明了有希望的结果

##### 为了估计模糊图（散焦图或视差图），以前的工作要么需要额外的数据，例如用于监督信号的合成数据，要么需要预校准的模糊内核

##### 最近，对比语言图像预训练框架（CLIP）在语义分割、对象检测和3D点云理解等视觉任务中取得了巨大的成功

① 通过探索 CLIP 在低级视觉任务中的潜力，我们提出了一种语言驱动的DP ( LDP ) 离焦去模糊框架。 ② 我们根据 DP 对的模糊和视差之间的几何关系设计了一种用于模糊图估计的图像文本格式。 ③ 我们提出了模糊优先注意（BPA）块、模糊加权损失和模糊感知损失，以鼓励 DP 对的锐利恢复。

## 4 其它

### 4.1 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
