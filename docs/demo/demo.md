# 重参数化面面观



## 导引

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqa4hpccfgj30hi0a3mxh.jpg)

## 什么是重参数

借用[丁霄汉](https://zhuanlan.zhihu.com/p/361090497)博士的说法：“结构A对应一组参数X，结构B对应一组参数Y，如果我们能将X等价转换为Y，就能将结构A等价转换为B”。也就是，如果结构A是训练阶段的复杂结构，结构B是推理阶段的精简结构，训练阶段的结构A可以极大的丰富模型的微观结构，进而提升模型性能，但对于部署不够友好；与结构A等价的结构B更为精简，且能取得同等性能，部署更友好，但从头训练时因缺乏丰富的围观结构导致性能不佳。

那么，什么样的结构可以进行这样的转换呢？如果连续的两个操作均为线性变换，且两者中间未插入非线性激活，那么这两个操作就可以合并为一个等价的线性变换。假设输入为x，连续的两个线性变换为A和B，那么输出 $y = B(Ax)$ ；如果我们可以构造一个 $C=BA$ ，那么就有$y = B(Ax) = Cx$。

那么，什么样的操作时线性变换呢？在深度学习中，卷积、BatchNorm、全连接层均为线性操作，像Conv+BN、Linear+BN的推理融合已成为一种非常基本的推理优化技巧。



## 基础知识

在介绍具体方法之前，我们先来介绍一些关于卷积融合的基础知识。无论是Conv+BN的融合，还是Conv+Conv的融合，它们能融合的根本原因在于：**操作的线性**特性。

### 卷积的线性

我们假设卷积输入通道数为C，输出通道数为D，卷积尺寸为$K \times K$，那么其卷积核为$F \in R^{D \times C \times K \times K}$，偏置参数$b \in R^D$可选。为方便后续的合并，我们将偏置参数表示为$REP(b) \in R^{D \times H \times W}$，此时卷积可以表示如下：
$$
O = I * F +REP(b)
$$
输出通道的每个值可以表示如下：
$$
O_{j,h,w} = \sum_{c=1}\sum_{u=1}\sum_{v=1} F_{j,c,u,v} X(c,h,w)_{u,v} + b_j
$$
卷积的线性特性可以从上述沟通推导得到，包含同质特性与加法特性：
$$
I*(pF) = p(I*F), \forall p \in R  \\
I*F^{(1)} + I*F^{(2)} = I*(F^{(1)} + F^{(2)})
$$
注：加法特性只有当两个滤波器满足相同配置(比如通道数量、卷积核尺寸、stride、padding等等)时才满足。

### Conv+BN融合

一般而来，卷积与BN这对“基友”会同时出现，而两者在推理时又可以合并为单一卷积。那么如何合并呢？这个比较简单，公式如下：
$$
O_{j,:,:} = ((I*F)_{j,:,:} - \mu_j) \frac{\gamma_j}{\sigma_j} + \beta_j
$$
这里利用了卷积的同质特性进行合并。变换后的卷积参数如下：
$$
F^{'}_{j,:,:,:} \leftarrow \frac{\gamma_j}{\sigma_j} F_{j,:,:,:}  \\
b^{'}_{j} \leftarrow - \frac{\mu_j \gamma_j}{\sigma_j} + \beta_j
$$

### 双分支合并

分支合并利用了卷积的加法特性，两个卷积的参数合并方式如下：
$$
F^{'} \leftarrow F^{(1)} + F^{(2)} \\
b^{'} \leftarrow b^{(1)} + b^{(2)}
$$
在这方面比较典型的网络是ACNet中的$1\times 3, 3\times 1, 3\times 3$等分支卷积的合并。

### 双分支Concat

除了Add融合外，Concat进行多分支的融合是另一种最常见的方式。Concat方式的公式也是非常的简单，描述如下：
$$
CONCAT(I*F^{(1)} + REP(b^{(1)}), I*F^{(2)} + REP(b^{(2)})) = I*F^{'} + REP(b^{'})
$$
此时$F^{'} \in R^{(D_1 +D_2) \times C \times K \times K}, b^{'} \in R^{D_1 +D_2}$。这里的组合是一种通用变换，可以将两个分之的卷积变换成组卷积形式。注：在这个情况下，该分支的卷积序列应当具有相同的groups参数。

这种情况的一个典型模块为SqueezeNet中`Fire`模块，它由两路并行分支构成(一个分支为$1\times 1$，一个分支为$3\times 3$)，最后通过concat进行融合。

### 并行卷积合并

我们可以将连续的$1\times 1$卷积与$K \times K$卷积合并为单个$K \times K$卷积；类似的，我们也可以将$K \times K$卷积与$1\times 1$卷积合并为单个$K \times K$卷积此时$1\times 1$与$K \times K$两个卷积的卷积核分别为$D\times C \times 1 \times1$与$E\times D \times K \times K$(注：D可以是任意整数，所以这里可以极大的提升FLOPs，比如把D设置为10000，哈哈)。这两个卷积核的合并方法也是非常简单，公式如下：
$$
F^{'} \leftarrow F^{(2)} * TRANS(F^{(1)})
$$
注：$TRANS(F^{(1)})$表示将$F^{(1)}$为$R^{C\times D \times 1 \times 1}$空间。

### 多分支、多尺度卷积合并

考虑到$k_h \times k_w(k_h \le K, k_w \le K)$可以通过零padding等价变换为$K \times K$卷积，因此$1\times1, 1\times K, K\times 1$可以等价的转换，这也就是ACNetV1的本质所在。谷歌提出的MixConv以及斯坦福大学提出的SqueezeNet中的Fire模块可视作这种情形下的一种特殊形式。

### 典型操作

上面主要从一些通用场景进行介绍，实际应用场景中可能还包含一些特殊的操作，比如均值池化、depthwise卷积、组卷积、Identity(可视作参数固定的$1\times 1$卷积)等。

对于这些典型的操作，我们可以先将其转换为标准卷积性，再进行卷积的融合。事实上，我们还可以进行depthwise卷积+标准卷积的融合、标准卷积+组卷机的融合以重点突出近邻通道之间的相关性。

另外，需要指出的是卷积是全连接层的一个特例，即参数共享+稀疏形式的全连接层。因此，我们可以在某些特定条件下采用卷积对全连接层进行增广，比如最近提出的RepMLP。RepMLP是在ResNet的stage部分对bottleneck进行的操作，RepMLP理论上也可以在classifier部分进行增强处理。

当然，了解了上述原理后，大家就可以随机发挥了，只要等价并且能涨点就可以了。



## 方法介绍

### ExpandNet

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqa5z9c8utj30pq0osq5y.jpg" style="zoom: 33%;" />

上图给出了ExpandNet一文的重参数结构示意图，它采用了串联方式进行结构重参数。比如，采用$1\times 1, k\times k, 1\times 1$的组合最大程度的提升参数量以及模型性能，在推理阶段可以折叠为简单的$k\times k$卷积。

需要注意的是：在串联重参数方案中，如果期望折叠后的卷积核尺寸为k，那么串联的卷积尺寸不能大于k,如果其中存在一个为k，那么其他的只能设置为1。比如，我们期望折叠后的卷积核尺寸为3，那么，我们只能采用1-3-1的组合；如果我们期望折叠后的卷积核尺寸为5，我们可以选择1-5-1的组合，也可以选择3-3的组合。总而言之：**折叠前后的感受野必须一致**。



### ACNet

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqa5vu7xe7j31oc0cgagu.jpg)

上图给出了ACNet训练与推理时的结构示意图，它采用了并行方式进行结构重参数化。在推理阶段，它由$1\times 3, 3\times 1, 3\times 3$三种形式的卷积构成，这种配置用于提升中间“十”字骨架的权值。ACNet这种非对称结果在图像超分、目标检测等领域得到了应用，同时也出现在一些产品化方案中。

更详细介绍建议查看BBuf同学的解读：[无痛的涨点技巧：ACNet](https://zhuanlan.zhihu.com/p/131282789)

### DO-Conv

![](https://pic1.zhimg.com/80/v2-8eae167d0fdb23cda4558ae2960503a8_1440w.jpg)

上图给出了DO-Conv的两种计算流程图，一种是特征分解模式，一种kernel分解模式。建议采用kernel分解模式，这种方式在训练完成后可以转换为固定的卷积核，达到训练重参数、推理轻参数的目的。

更详细介绍建议查看52CV的解读：[DO-Conv无痛涨点：使用over-parameterized卷积层提高CNN性能](https://zhuanlan.zhihu.com/p/361260569)

### RepVGG

<img src="https://tva1.sinaimg.cn/large/006C3FgEgy1gml1egfws6j30fv0jagod.jpg" style="zoom: 50%;" />

上图给出了RepVGG中的重参数示意图，它采用了Identity、$1\times 1$卷积以及$3\times 3$卷积进行重参数设置，在完成训练后，我们可以轻易的将Identity、$1\times1,3\times 3$卷积三个分支合并为单个$3\times 3$卷积。

更详细介绍建议查看Happy的解读：[RepVGG|让你的ConVNet一卷到底，plain网络首次超过80%top1精度](https://mp.weixin.qq.com/s/M4Kspm6hO3W8fXT_JqoEhA)。



### DBB

![](https://tva1.sinaimg.cn/large/008eGmZEgy1goxjd2bwg6j31170gr41d.jpg)

上图给出了本文所设计的ＤＢＢ结构示意图。类似Inception,它采用$1\times1, 1\times 1-K\times K, 1\times 1-AVG$等组合方式对原始$K \times K$卷积进行增强。对于$1\times 1-K \times K$分支，我们设置中间通道数等于输入通道数并将$1\times 1$卷积初始化为Identity矩阵；其他分支则采用常规方式初始化。此外，在每个卷积后都添加BN层用于提供训练时的非线性，这对于性能提升很有必要。

更详细介绍建议查看Happy的解读：[CVPR2021|“无痛涨点”的ACNet再进化，清华大学&旷视科技提出Inception类型的DBB](https://mp.weixin.qq.com/s/F7LMjuUuWnuQX1SJ5C5u6Q)。

### RepMLP

<img src="https://files.mdnice.com/user/306/f9fdac5c-0546-467f-b624-ab7230f90952.png" style="zoom:50%;" />

上图给出了RepMLP的结构示意图，它能同时利用卷积层的局部结构提取能力与全连接层的全局建模、位置感知特性。更详细的介绍建议查看Happy的解读： [“重参数宇宙”再添新成员：RepMLP，清华大学&旷视科技提出将重参数卷积嵌入到全连接层](https://mp.weixin.qq.com/s/UgIFmWJsUfTtZgt1yznt9w)。

### PSConv

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqbt5w8bcxj30ob09zabh.jpg)

上图为PSConv的示意图，它将多尺度卷积纳入到统一的架构中。上图可能不太容易看懂，我们来看下图，不同颜色代表不同的感受野。比如，红色代表$3\times 3$,蓝色代表$5\times 5$， 绿色代表$7\times 7$， 黄色代表$9\times 9$。当把所有的卷积核参考ACNet中方式padding到$9\times 9$后，是不是发现：PSConv就变为了简简单单的$9\times 9$卷积了。当然FLOPs也会变大不少。

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1ggqrkufp0ij30uu0hkmzf.jpg)

PSConv的实现中采用了$3\times3, 5\times 5, 7\times 7$三个尺度的感受野，如果按照重参数化方案进行转换的话，转换后的卷积核尺寸为$7\times 7$，计算量会进一步加剧，且现有框架对$3\times 3$以外的卷积优化程度不够，所以会导致速度变慢。如果我们固定转换后的卷积核尺寸是$3\times 3$的话，我们可以设计不同颜色区域采用不同的卷积：$1\times 1, 3\times 1, 1\times 3, 3\times 3$。这样是不是就可以达到折叠的目的了，哈哈。



## 重参数变种

到此为止，我们基本上已经把业界知名的重参数化方案进行了简要性的介绍。如果有哪位同学想进行更多样的结构设计可参考我们很早之前的一篇文章[稀疏卷积在模型设计中的应用](https://zhuanlan.zhihu.com/p/76829900)结合上述提到的折叠原理进行更多样性的模块设计。

比如下面几种简单的特例，注：这里仅列出了笔者与小伙伴们在这方面探索时几个比较有用的组合，但这里的可尝试空间还是非常大的。

-   结合了ResNet、DBB等不同的折叠方式。

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqbt63f2k3j30bk087wek.jpg)

-   结合ExpandNet、ACNet、DBB的模块。

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqbt6782cdj30g4084t8u.jpg)

-   结合Res2Net、ACNet等思想的模块。

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqbt6axgz2j30b008hwem.jpg)

-   结合PSConv与ACNet而设计的模块。

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqbt6e7pc6j30b1077t8q.jpg)

-   结合DenseNet与ACNet而设计的模块。

![](https://tva1.sinaimg.cn/large/008i3skNgy1gqbt6ijc1mj30az06b3yj.jpg)



## 应用场景

从笔者有限的了解来看，ACNet、ExpandNet的重参数思想在图像超分、图像去摩尔纹、目标检测中得到了不少应用，比如

-   [ACNet|增强局部显著特征，哈工大左旺孟老师团队提出非对称卷积用于图像超分](https://mp.weixin.qq.com/s/zzS16Zxc4WXbXtnH70uBnQ)
-   [46FPS+1080Px2超分+手机NPU，arm提出一种基于重参数化思想的超高效图像超分方案](https://mp.weixin.qq.com/s/yXLEI2OFwoTwc3VDQBrv1A)
-   NTIRE的竞赛中曾有参赛者将ACNet与IMDN进行了组合，能显著提升模型的PSNR指标；
-   IJCAI2020卡通人脸检测竞赛的冠军方案采用了ACNet思想；
-   NTIRE2020图像去摩尔纹竞赛3nd的方案采用了ACNet思想；
-   NTIRE2020图像降噪的4th方案同样采用ACNet思想；
-   ......



## 我司应用

截止目前，重参数化方案已成为视频增强项目的必选模块，在压缩视频超分、FOV画质增强中均有应用，且能取得显著的性能、视觉效果提升。

关于重参数化方案在相关项目中的应用，由于涉及项目细节，这里略过。如果感兴趣可与视频增强组一同讨论。

重参数化有这样几个优势：

-   在训练过程中引入多分支可以改善网络训练过程中的梯度消失问题；
-   在训练过程中引入不同尺度分支可以提升模型的特征提取能力；
-   大容量有助于提升模型性能，而其可折叠特性使其可以做到推理无耗时增加；
-   可以一定程度上提升模型的客观指标。



## 参考文献

-   On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization. Sanjeev Arora, Nadav Cohen, Elad Hazan. ([arxiv](https://arxiv.org/abs/1802.06509)), ([code]()), ([解读]())
-   ExpandNets: Linear Over-parameterization to Train Compact Convolutional Networks. Shuxuan Guo, Jose M. Alvarez, Mathieu Salzmann. ([arxiv](https://arxiv.org/abs/1811.10495)), ([code](https://github.com/GUOShuxuan/expandnets)), ([解读]())
-   ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks. Xiaohan Ding, Yuchen Guo, Guiguang Ding, Jungong Han. **ICCV, 2019**  ([arxiv](https://arxiv.org/pdf/1908.03930.pdf))([code](https://github.com/DingXiaoH/ACNet)) ([解读](https://zhuanlan.zhihu.com/p/131282789))
-   DO-Conv: Depthwise Over-parameterized Convolutional Layer. Jinming Cao, Yangyan Li, Mingchao Sun, Ying Chen, Danii Lischinski, Daniel Cohen-Or, Baoquan Chen, Changhe Tu. ([arxiv](https://arxiv.org/abs/2006.12030)), ([code](https://github.com/yangyanli/DO-Conv)), ([解读]())
-   PSConv: Squeezing Feature Pyramid into One Compact Poly-Scale Convolutional Layer. Duo Li, Anbang Yao, Qifeng Chen. **ECCV 2020**. ([arxiv](https://arxiv.org/abs/2007.06191)), ([code](https://github.com/d-li14/PSConv)), ([解读](https://mp.weixin.qq.com/s/pKY0tup88wfFMEX4PWfOzA))
-   RepVGG: Making VGG-style ConvNets Great Again. Xiaohao Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun. **CVPR 2021**  ([arxiv](httsp://arxiv.org/abs/2101.03697)), ([code](https://github.com/DingXiaoH/RepVGG)), ([解读](https://mp.weixin.qq.com/s/M4Kspm6hO3W8fXT_JqoEhA))
-   Diverse Branch Block: Building a Convolution as an Inception-like Unit. Xiaohao Ding, Xiangyu Zhang, Jungong Han, Guiguang Ding. **CVPR 2021**  ([arxiv](https://arxiv.org/pdf/2103.13425.pdf)), ([code](https://github.com/DingXiaoH/DiverseBranchBlock)), ([解读](https://mp.weixin.qq.com/s/F7LMjuUuWnuQX1SJ5C5u6Q))
-   RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition. Xiaohan Ding, Xiangyu Zhang, Jungong Han, Guiguang Ding. ([arxiv](https://arxiv.org/abs/2105.01883)), ([code](https://github.com/DingXiaoH/RepMLP)), ([解读](https://mp.weixin.qq.com/s/UgIFmWJsUfTtZgt1yznt9w))





