# 基于冻结的 NeurIPS 2025 分子生成与分子设计论文语料，综合模型家族、分子表示、条件控制、评测证据、局限与研究机会，形成可追溯的中文综述。

## 执行摘要

冻结语料呈现四条相互交织的主线：MOLTD以12次函数评估在QM9报告99.40%的原子稳定性；PAFlow在Binding MOAD无额外训练时报告Vina Score -9.12和83.4%的高亲和率；SCENT把合成成本、模式发现与回溯合成成功率纳入生成目标；SLDM则报告相对EDM最高100倍的采样加速。由于数据、指标、基线版本及协议哈希普遍缺失，这些论文内部结果不可直接比较，不能据此建立统一排行榜。[@paper-6b0b363aaa775ce5b7c0f4b3f6af7925] [@paper-227836fa5cb95d6b8146851c69fa3ed5] [@paper-0c4a0e599ab95669bedcdf02c194d35b] [@paper-c951acbd919f5885ab836cc2652d3efd]

## 研究范围与方法

本报告面向分子机器学习研究者，范围限定为NeurIPS 2025中以生成、设计、编辑或采样分子及原子结构为核心贡献的会议论文。抽取范围：full_pdf=15；全文、摘要和元数据证据已分层，缺失全文不作全文事实表述。综合仅使用冻结证据对象，不补写未登记的比较组、协议、数值或参考文献。

## 检索流与语料画像

会场检索共发现并去重筛查5823条记录，排除5808条，最终纳入15篇且均获得全文；15篇全部来自新发现队列，基础性种子、引文滚雪球和用户文库来源均为0。规范出版状态为同行评审12篇、状态未知3篇，预印本和workshop均为0。该流量图仅描述冻结的会场检索，不代表跨数据库或历史文献的完整覆盖。

## 领域图景与分类体系

方法分类可按表示与动力学划分：DMol在离散分子图上耦合节点—边噪声并压缩基序；DiTMC将图条件、位置编码和条件流匹配接入扩散Transformer；能量扩散工作用Fokker–Planck正则约束生成采样与Langevin模拟的一致性；FuncBind则以全原子神经场支持跨小分子、抗体CDR和大环肽的结构条件生成，并给出抗体体外结合验证。[@paper-932bf5002d015ba99e4ff4b3eb2fe4cf] [@paper-d7efc018e794559dbcc5c8bd0246fcb9] [@paper-79fa03d04ea15f028e3e58a47b374b33] [@paper-f333918e7f6f5ec197053046bbfc7bfb]

## 分子生成模型证据综合

证据显示，生成质量取决于表示、条件机制和数据复杂度。多目标强化学习扩散模型在QM9、ZINC15和PubChem分别报告98.17±0.07%、99.02±0.46%和16.23±9.72%的有效率，提示跨数据集数值不可直接比较且复杂化学空间仍是瓶颈。UDM-3D在GEOM-Drugs和QM9重构中均报告100%原子与键准确率，坐标RMSD分别为0.0008 Å和0.0002 Å。CIDD的CoT配置在CrossDocked2020报告SA 0.735、MRR 81.74%和成功率34.59%。[@paper-5de4af5c958a5ab693fe71abc1961e10] [@paper-71f47a1543d058e3850ee1c543c5943f] [@paper-76c14309115452ebb0f2eb41b2173a02]

## 方法、数据集、基准与资源对比

方法与实验协议可比性矩阵应同时记录任务、数据集、采样预算、误差定义和硬件条件。JAMUN的同架构实验显示，walk-jump在3149个样本时使用6298次评估、JSD为0.1501，而全扩散使用399923次评估、JSD为0.1363，体现速度—分布保真折衷；SLDM的100倍加速则来自NFE口径。两项结论因注册化协议及成对预算信息不完整而不可直接比较。[@paper-545719f013a6564ea169585b1df47ebb] [@paper-c951acbd919f5885ab836cc2652d3efd]

## 矛盾结论、不可比项与证据局限

FMA-PO+与FitDock、Vina在AlignDockBench上的结果形成分歧，但注册映射缺失，不能据此做确定性排序。其主结果为1.62±1.33 Å及77.78%的RMSD低于2 Å比例，但所有相关结果仍不可直接比较。CBYG的回溯合成指标与约半数分子仍不可合成之间存在冲突。CHEF NMR在总体top-10表现与超大分子失败率之间存在冲突。后两项分别表明代理合成指标不能消除实际可合成性缺口，以及总体平均性能不能覆盖复杂度尾部风险。[@paper-6838bd00496350ea93c919bb5a3af4d6] [@paper-3b243975c39c51a2b7e8385b26ef5691] [@paper-38e0375ff7fa527090876cc4301a330d]

## 研究空白与可检验机会

最直接的可检验机会，是在更复杂分子上分离架构容量、奖励模型校准与数据覆盖三类因素。强化学习扩散模型在PubChem仅报告16.23±9.72%的有效率，而QM9和ZINC15均接近99%；这些跨数据集结果不可直接比较，但足以支持预注册分子大小分层、统一生成预算、候选级ADMET公开和独立湿实验验证。[@paper-5de4af5c958a5ab693fe71abc1961e10]

## 实践结论与建议

实践上应把生成质量拆成结构有效性、目标相互作用、合成可行性与实验转化四层：CIDD适合研究结构交互与语言模型推理的协作；PAFlow提供口袋条件和跨数据集无再训练测试；SCENT显式优化合成成本与回溯合成成功；FuncBind展示了从多模态全原子生成走向SPR验证的路径。选择方法时应优先复现论文内协议，并避免把这些异构证据合并成单一总分。[@paper-76c14309115452ebb0f2eb41b2173a02] [@paper-227836fa5cb95d6b8146851c69fa3ed5] [@paper-0c4a0e599ab95669bedcdf02c194d35b] [@paper-f333918e7f6f5ec197053046bbfc7bfb]

## 本报告限制与更新状态

本报告是15篇近期论文的冻结横截面，其中同行评审12篇、出版状态未知3篇；没有基础性种子，因而不能表达领域历史演进。检索虽在既定会场来源内完成，但不覆盖其他会议、期刊、预印本或后续更正。多数论文缺少版本化数据集、基线、指标定义和协议哈希，故跨论文比较总体证据不足。

## 参考文献与附录

*规范参考文献由本地协调器根据冻结的 canonical metadata 生成；附录保留查询清单、排除原因、覆盖台账与主张台账。*

## 参考文献

[@paper-0c4a0e599ab95669bedcdf02c194d35b] Piotr Gaiński, Oussama Boussif, Andrei Rekesh, Dmytro Shevchuk, Ali Parviz, Mike Tyers, Robert A. Batey, Michał Koziarski. Scalable and Cost-Efficient de Novo Template-Based Molecular Generation. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/333581887bf483296118a97773cab0c1-Abstract-Conference.html
[@paper-227836fa5cb95d6b8146851c69fa3ed5] Jingyuan Zhou, Hao Qian, Shikui Tu, Lei Xu. Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/afe1aa79e5eea7955f553c61a307273e-Abstract-Conference.html
[@paper-38e0375ff7fa527090876cc4301a330d] Ziyu Xiong, Yichi Zhang, Foyez Alauddin, Chu Xin Cheng, Joon An, Mohammad Seyedsayamdost, Ellen D. Zhong. Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html
[@paper-3b243975c39c51a2b7e8385b26ef5691] Seungyeon Choi, Hwanhee Kim, Chihyun Park, Dahyeon Lee, Seungyong Lee, Yoonju Kim, Hyoungjoon Park, Sein Kwon, Youngwan Jo, Sanghyun Park. Controllable 3D Molecular Generation for Structure-Based Drug Design Through Bayesian Flow Networks and Gradient Integration. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/b6d0df730c5976ad918bbf4fb30afe7d-Abstract-Conference.html
[@paper-545719f013a6564ea169585b1df47ebb] Ameya Daigavane, Bodhi Vani, Darcy Davidson, Saeed Saremi, Joshua Rackers, Joseph Kleinhenz. JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensemble Generation. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/3476352269e5a76b91cb4670390f1b5c-Abstract-Conference.html
[@paper-5de4af5c958a5ab693fe71abc1961e10] Lianghong Chen, Dongkyu Kim, Mike Domaratzki, Pingzhao Hu. Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/b4a804ef8f138e301680f6dda6253be4-Abstract-Conference.html
[@paper-6838bd00496350ea93c919bb5a3af4d6] Noémie Bergues, Arthur Carré, Paul Join-Lambert, Brice Hoffmann, Arnaud Blondel, Hamza Tajmouati. Template-Guided 3D Molecular Pose Generation via Flow Matching and Differentiable Optimization. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html
[@paper-6b0b363aaa775ce5b7c0f4b3f6af7925] Zhilong Zhang, Yuxuan Song, Yichun Wang, Jingjing Gong, Hanlin Wu, Dongzhan Zhou, Hao Zhou, Wei-Ying Ma. Accelerating 3D Molecule Generative Models with Trajectory Diagnosis. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/319e96498360f5f6f8f21623b509b8ca-Abstract-Conference.html
[@paper-71f47a1543d058e3850ee1c543c5943f] Yanchen Luo, ZHIYUAN LIU, Yi Zhao, Sihang Li, Hengxing Cai, Kenji Kawaguchi, Tat-Seng Chua, Yang Zhang, Xiang Wang. Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/2c2f104b95306ab94058a97ba13b8927-Abstract-Conference.html
[@paper-76c14309115452ebb0f2eb41b2173a02] Bowen Gao, Yanwen Huang, Yiqiao Liu, Wenxuan Xie, Bowei He, Haichuan Tan, Wei-Ying Ma, Ya-Qin Zhang, Yanyan Lan. CIDD: Collaborative Intelligence for Structure-Based Drug Design Empowered by LLMs. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/14391f8ff68b7c51314f7897c581af9c-Abstract-Conference.html
[@paper-79fa03d04ea15f028e3e58a47b374b33] Michael Plainer, Hao Wu, Leon Klein, Stephan Günnemann, Frank Noe. Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/231be94eaf8dcbc49a95b256c9b6b8b5-Abstract-Conference.html
[@paper-932bf5002d015ba99e4ff4b3eb2fe4cf] Peizhi Niu, Yu-Hsiang Wang, Vishal Rana, Chetan Rupakheti, Abhishek Pandey, Olgica Milenkovic. DMol: A Highly Efficient and Chemical Motif-Preserving Molecule Generation Platform. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/e70caa88a65dd6cbc227f66e58211f18-Abstract-Conference.html
[@paper-c951acbd919f5885ab836cc2652d3efd] Yuyan Ni, Shikun Feng, Haohan Chi, Bowen Zheng, Huan-ang Gao, Wei-Ying Ma, Zhi-Ming Ma, Yanyan Lan. Straight-Line Diffusion Model for Efficient 3D Molecular Generation. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/2d0842550e6d92b0e27e7e810b1a4792-Abstract-Conference.html
[@paper-d7efc018e794559dbcc5c8bd0246fcb9] J. Thorben Frank, Winfried Ripken, Gregor Lied, Klaus-Robert Müller, Oliver Unke, Stefan Chmiela. Sampling 3D Molecular Conformers with Diffusion Transformers. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/f6fbaf131fa71b036d7b8c49379f3af4-Abstract-Conference.html
[@paper-f333918e7f6f5ec197053046bbfc7bfb] Matthieu Kirchmeyer, Pedro O O. Pinheiro, Emma Willett, Karolis Martinkus, Joseph Kleinhenz, Emily Makowski, Andrew Watkins, Vladimir Gligorijevic, Richard Bonneau, Saeed Saremi. Unified all-atom molecule generation with neural fields. NeurIPS 2025, 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/64fd109ea88e4bd6876806038d725740-Abstract-Conference.html
