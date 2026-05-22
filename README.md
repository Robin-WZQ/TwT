# 👿 Trigger without Trace: Towards Stealthy Backdoor Attack on Text-to-Image Diffusion Models

> [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

We propose TwT, an attack method based on **syntactic structures** that exhibits **strong resistance to advanced detection methods**.

## 🔥 News
- [2026/05/07] Our work has accepted by TIFS!🎉🎉🎉

## 👀 Overview

<div align=center>
<img src='https://github.com/Robin-WZQ/TwT/blob/main/Visualization/Models.png' width=700>
</div>

our approach leverages syntactic structures as backdoor triggers to amplify the sensitivity to textual variations, effectively breaking down the semantic consistency. Besides, a regularization method based on Kernel Maximum Mean Discrepancy (KMMD) is proposed to align the distribution of cross-attention responses between backdoor and benign samples, thereby disrupting attention consistency. 

## 🧙‍♂️ Trigger without Trace

<div align=center>
<img src='https://github.com/Robin-WZQ/TwT/blob/main/Visualization/Assimilation%2520Phenomenon.png' width=600>
</div>

The visualization of cross-attention maps during image generation. TwT generates attacker specified images while effectively mitigating "Assimilation Phenomenon".

<div align=center>
<img src='https://github.com/Robin-WZQ/TwT/blob/main/Visualization/against_ufid.png' width=600>
</div>

Our method accurately recognizes specific syntax, effectively avoiding been identified by pertubation-based method, i.e., UFID. Syntax trigger here is "(DET)(NOUN)(ADP)(DET)(NOUN)(VERB)(ADP)(NOUN)".

## 🧭 Getting Start

TwT has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n TwT python=3.10
   conda activate TwT
   python -m pip install --upgrade pip
   pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

2. Then you can install required packages thourgh:

   ```
   pip install -r requirements.txt
   ```

## 🏃🏼 Running Scripts

#### Backdoor Injection

- Inject one backdoor w/o pretrained model

```CUDA_VISIBLE_DEVICES=0,1 python backdoor_injection_main.py \
 CUDA_VISIBLE_DEVICES=0,1 python backdoor_injection_main.py
     -c './configs/backdoor_invisible/backdoor_1.yaml' \
     -l 1e-2 \
     -t './data/train/backdoor_1.txt'\
     -p False
```

- Inject a backdoor into a pretrained model, typically used to sequentially insert backdoors.

```CUDA_VISIBLE_DEVICES=0,1 python backdoor_injection_main.py \
 CUDA_VISIBLE_DEVICES=0,1 python backdoor_injection_main.py
     -c './configs/backdoor_invisible/backdoor_1.yaml' \
     -l 1e-2 \
     -t './data/train/backdoor_1.txt' \
     -p True \
     -pp './results/backdoor_1/'
```

**Checkpoints**

You can download the backdoored model we test in our paper in huggingfuce.

|    ID     | Link |
| :-------: | :--: |
| backdoor1 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_5_a_blond)   |
| backdoor2 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_8_a_man)   |
| backdoor3 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_15_the_effiel)   |
| backdoor4 |   [[link]](https://huggingface.co/RobinWZQ/backdoor_KMMD_len_20_a_motor)   |

For more types of backdoored model, please refer to [models](https://huggingface.co/RobinWZQ).


#### Evaluation

- FID (Frechet Inception Distance)

```
# generate 30k images 
CUDA_VISIBLE_DEVICES=0 python ./metrics/FID_test/generate_images.py --backdoor_model backdoor_1 --epoch 599

# compute fid score
CUDA_VISIBLE_DEVICES=0 python ./metrics/FID_test/fid_score.py --path1 ./coco_val.npz --path2 ./backdoor_1/599
```

- ASR (Attack Success Rate)

```
CUDA_VISIBLE_DEVICES=0 python ./metrics/ASR_test/generate_images_asr.py --backdoor_model backdoor_1 --epoch 599
```

- DSR (Detect Success Rate)

> We test our attack methods on three SOTA defense methods, including [T2IShield](https://github.com/Robin-WZQ/T2IShield) and [UFID](https://github.com/GuanZihan/official_UFID).

```
# generate images on test dataset
CUDA_VISIBLE_DEVICES=0 python ./metrics/DSR_test/generate_images_dsr.py --backdoor_model backdoor_1 --epoch 599

# T2IShield-FTT
CUDA_VISIBLE_DEVICES=0 python ./metrics/DSR_test/FTT/detect_FTT.py

# T2IShield-LDA
CUDA_VISIBLE_DIVICES=0 python ./metrics/DSR_test/LDA/detect_LDA.py

# UFID
run UFID_test.ipynb
```

## 🔨 Results
- TwT achieves an ASR of 97.5%. More results can be found in the paper.

<div align=center>
<img src='https://github.com/Robin-WZQ/TwT/blob/main/Visualization/results.png' width=600>
</div>

- Here we show some qualitative results of TwT. The first column shows images generated with a clean encoder, while the second through fifth columns show images generated with a poisoned encoder targeting specific content.
> Trigger syntax below: (DET)(NOUN)(ADP)(DET)(NOUN)(VERB)(ADP)(NOUN)

<div align=center>
<img src='https://github.com/Robin-WZQ/TwT/blob/main/Visualization/Examples.png' width=400>
</div>

## 📄 Citation

If you find this project useful in your research, please consider cite:
```
@ARTICLE{11527385,
  author={Zhang, Jie and Wang, Zhongqi and Shan, Shiguang and Chen, Xilin},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Trigger without Trace: Towards Stealthy Backdoor Attack on Text-to-Image Diffusion Models}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Diffusion models;Text to image;Syntactics;Training;Automatic speech recognition;Conferences;Computers;Toxicology;Computer vision;Backdoor Attack;Text-to-Image Diffusion Models;Syntactic Trigger},
  doi={10.1109/TIFS.2026.3695430}}
```
🤝 Feel free to discuss with us privately!
