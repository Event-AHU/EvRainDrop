# Event_RainDrop

<!--### News--> 


### Abstract 
Event cameras produce asynchronous event streams that are spatially sparse yet temporally dense. Mainstream event representation learning algorithms typically use event frames, voxels, or tensors as input. Although these approaches have achieved notable progress, they struggle to address the undersampling problem caused by spatial sparsity. In this paper, we propose a novel hypergraph-guided spatio-temporal event stream completion mechanism, which connects event tokens across different times and spatial locations via hypergraphs and leverages contextual information message passing to complete these sparse events. The proposed method can flexibly incorporate RGB tokens as nodes in the hypergraph within this completion framework, enabling multi-modal hypergraph-based information completion. Subsequently, we aggregate hypergraph node information across different time steps through self-attention, enabling effective learning and fusion of multi-modal features. Extensive experiments on both single- and multi-label event classification tasks fully validated the effectiveness of our proposed framework. The source code of this paper will be released upon acceptance. 

### Configuration 
Install env
``` bash
conda create -n evraindrop python=3.9
conda activate evraindrop
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

<!--### ### Framework --> 


### Training and Testing 
``` bash
train & test
bash train.sh
```


<!--### Experimental Results--> 



### Acknowledgement 

* Zhang, Xiang, et al. ["Graph-Guided Network for Irregularly Sampled Multivariate Time Series."](https://arxiv.org/pdf/2110.05357) International Conference on Learning Representations 2022 
  [[Paper](https://arxiv.org/pdf/2110.05357)]
  [[Code](https://github.com/mims-harvard/Raindrop)] 

* [EventPAR Dataset] ["RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset andAn Asymmetric RWKV Fusion Framework."](https://arxiv.org/abs/2504.10018) arXiv 2025
  [[Paper](https://arxiv.org/abs/2504.10018)]
  [[Code](https://github.com/Event-AHU/OpenPAR/tree/main/EventPAR_Benchmark)] 

* [DUKE PAR Dataset] ["Performance Measures and a Data Set forMulti-Target, Multi-Camera Tracking."](https://arxiv.org/abs/1609.01775) European Conference on Computer Vision 2016
  [[Paper](https://arxiv.org/abs/1609.01775)]

* [MARS PAR Dataset] ["Mars: A video benchmark for large-scale person re-identification."](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_52) European Conference on Computer Vision 2016
  [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_52)]

* [PokerEvent Dataset] ["SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition."](https://arxiv.org/abs/2308.04369) Transactions on Cognitive and Developmental Systems 2025
  [[Paper](https://arxiv.org/abs/2308.04369)]
  [[Code](https://github.com/Event-AHU/SSTFormer)] 

* [HARDVS Dataset] ["HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors."](https://arxiv.org/abs/2211.09648) Association for the Advancement of Artificial Intelligence 2024
  [[Paper](https://arxiv.org/abs/2211.09648)]
  [[Code](https://github.com/Event-AHU/HARDVS)] 



### Citation 
If you have any questions about this work, please leave an issue. Also, please give us a star if you think this paper helps your research.
