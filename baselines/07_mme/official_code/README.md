# MME
MME: Mitigating the Impact of Malware Evolution on API Sequence-based Windows Malware Detectors

Dataset URL: https://pan.baidu.com/s/1x9IzfgujqSx4vywWjwBsOQ?pwd=1111  Extract Code：1111

contact email:talentedyuan@gmail.com; weixinguan@iie.ac.cn

# MME Knowledge Graph For Windows API
MME Knowledge Graph
URL: https://pan.baidu.com/s/1JssXJPvoUmqZhiuRlthcOQ?pwd=2222 Extract Code:2222

# About the Dataset
Training Dataset：
- 2017_data.npz: Training set of embeddings for the original model
- Evolve_2017_data: Training set of embeddings for the enhanced model

Testing Dataset：
The embedding test set of the original model：
- final_2018_data.npz
- final_2019_data.npz
- final_2020_data.npz
- final_2021_data.npz

embedding test set of the enhanced model：
- final_evolve_2018_data.npz
- final_evolve_2019_data.npz
- final_evolve_2020_data.npz
- final_evolve_2021_data.npz

# How to read: 
train_data = np.load('./data/Evolve_2017_data.npz')

train_x = train_data['x']

train_y = train_data['y']

train_y_family = train_data['y_family']


train_data = np.load('./data/2017_data.npz')

train_x = train_data['x']

train_y = train_data['y']


Cite:
@ARTICLE{11269877,
  author={Wei, Xingyuan and Li, Ce and Lv, Qiujian and Li, Ning and Sun, Degang and Wang, Yan},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Mitigating the Impact of Malware Evolution on API Sequence-Based Windows Malware Detectors}, 
  year={2026},
  volume={21},
  number={},
  pages={45-60},
  keywords={Malware;Semantics;Knowledge graphs;Vectors;Long short term memory;Feature extraction;Contrastive learning;Computational modeling;Aging;Detectors;API sequence-based malware detection;malware evolution;API knowledge graph;contrastive learning;deep learning},
  doi={10.1109/TIFS.2025.3637727}}

