# 1.介绍

​    这是一个基于深度学习的端到端的异常入侵检测模型。思路是将网络流量数据按照五元组（源ip，目的ip，协议，源端口，目的端口），（目的ip，源ip，协议，目的端口，源端口）进行分割，将两个主机在会话中产生的所有数据包归属为一个样本，称之为会话样本。我没有采取自己在网络中收集流量的方式，而是选择了当前反映流行攻击趋势和具有较多种类的CICIDS2018S数据集，https://www.unb.ca/cic/datasets/ids-2018.html.

​	

​	在我的处理中，我按照统计学的规律对数据包个数和每个数据包的尺寸进行取舍，最后每个数据包的张量形式为：（32, 320)。分割脚本为trafficSplit中split.py文件。

  然后将数据包送入如下的网络模型中model中model_v2.py：

​    

![image-20220601144028690](C:\Users\alice\AppData\Roaming\Typora\typora-user-images\image-20220601144028690.png)

​        fig 1. the end to end detect model

这是一个结合cnn和rnn和注意力机制的网络模型，是不是和问答模型很像？这确实如此，因为我的初始模型并没有注意力层，初始的设计模型结构结构很简答，如下所示model中的model_v1.py：

![image-20220601144336269](C:\Users\alice\AppData\Roaming\Typora\typora-user-images\image-20220601144336269.png)

​             fig 2. the simple end to end detect model

不同于基于人工特征的入侵检测模型，本模型的决策机制是难以让人理解的，我非常好奇这个基于深度学习的端到端的模型是怎么执行入侵检测分类的，我基于梯度积分提取了模型在每个攻击类中的全局积分梯度，对模型做出了可解释性研究，train中的integrated.py模型。

详情可以参考thesis文件下的论文。

# 2 .模块介绍





