## AIP: Adversarial Item Promotion

This repository releases the PyTorch implementation of "AIP" in our paper ["Adversarial Item Promotion: Vulnerabilities at the Core of Top-N Recommenders that Use Images to Address Cold Start"](https://arxiv.org/abs/1901.10332).

Figure below illustrates the mechanics of AIP attack. 

<p align="center">
<img src="/figures/diagram_1_adv_recsys.png" width="500">
</p>


### How to use the code:
#### 0. Prerequisites

We carry out experiments on Ubuntu 18.04 environemnt with 32 cpu cores and 32G memory. A GPU is essential for efficiency.

```
Python3
PyTorch 1.4.0
```

#### 1. Prepare datasets

Clone the AIP repository and download data.

```
git clone https://github.com/liuzrcc/AIP.git
cd AIP/data
bash download_data.sh
```
Please also download visual features and list of cold item items at this [link](??).

#### 2. Train first recommender BPR and second stage visual ranler DVBPR and VBPR


```
cd ./train/
python BPRtrain.py -data_set=amazon -gpu_id=0 -factor_num=64 -epoch=2000 -batch_size=4096 -lambda1=1e-3 -learning_rate=0.01 num_workers=6
python DVBPRtrain.py -data_set=amazon -gpu_id=0 -factor_num=100 -epoch=20 -batch_size=128 -lambda1=1e-3 -lambda1=1 -learning_rate=1e-4 num_workers=6
python VBPRtrain.py -data_set=amazon -gpu_id=0 -factor_num=100 -epoch=2000 -batch_size=512 -lambda1=1e-4 -learning_rate=1e-4 num_workers=4
```
Pre-trained models are available at this [link](??).


#### 3. Generate first stage candidate set and calculate visual ranker score for candidate set

```
python gen_candidate_set.py -task=BPR-DVBPR -data_set=amazon -gpu_id=0 -model_path=./models/ -score_path=./bpr_score_index/
python gen_candidate_set.py -task=VBPR -data_set=amazon -gpu_id=0 -model_path=./models/ -score_path=./bpr_score_index/
python gen_candidate_set.py -task=AlexRank -data_set=amazon -gpu_id=0 -model_path=./models/ -score_path=./bpr_score_index/
```
Pre-calculated models are available at this [link](??).


#### 4. Mount AIP attacks

Choose model (DVBPR, VBPR, AlexRank) to attack, and also choose attack methods (INSA or EXPA).

```
python mount_AIP.py -data_set=amazon -gpu_id=0 -model_to_attack=DVBPR -attack_type=INSA
```
Adversarial item images are save in `./adv_output/` by default.

#### 5. Evaluate of AIP attacks

```
python eval.py -model_to_eval=DVBPR -data_set=amazon -gpu_id=0 -adv_item_path=amazon_INSA
```
Evaluation results are saved in `./results/` by default.

### Visualization of attacks:

A t-SNE 2-D visualization of cooperative item and adversarial items are shown below:

![vis](/figures/tsne_DVBPR.png)


Please cite the following paper if you use AIP in your research.

      @inproceedings{pire2019,
      Author = {Zhuoran Liu and Martha Larson},
      Title = {Adversarial Item Promotion: Vulnerabilities at the Core of Top-N Recommenders that Use Images to Address Cold Start},
      Year = {2020},
      booktitle={},
      }
      
The copyright of all the images belongs to the image owners.
