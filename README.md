# IBA: Invisible Backdoor Attack on Text-to-Image Diffusion Models

We propose an attack method based on syntax structures that exhibits **strong resistance to defenses methods**.

## 👀 Overview

<div align=center>
<img src='https://github.com/Robin-WZQ/IBA/blob/main/images/Assimilation%20Phenomenon.png' width=800>
</div>

The visualization of cross-attention maps during image generation. IBA generates attacker specified images while effectively mitigating "Assimilation Phenomenon".

## 🧭 Getting Start

IBA has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n IBA python=3.10
   conda activate IBA
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
> Here we show some qualitative results of IBA. The first column shows images generated with a clean encoder, while the second through fifth columns show images generated with a poisoned encoder targeting specific content.
>
> Trigger syntax: (DET)(NOUN)(ADP)(DET)(NOUN)(VERB)(ADP)(NOUN)

<div align=center>
<img src='https://github.com/Robin-WZQ/IBA/blob/main/Visualization/results.png' width=800>
</div>

## 📄 Citation
If you find this project useful in your research, please consider cite:
```
@misc{zhang2024IBA,
      title={Invisible Backdoor Attack on Text-to-Image Diffusion Models}, 
      author={Jie Zhang and Zhongqi Wang and Shiguang Shan and Xilin Chen},
      year={2025},
      eprint={250x.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
🤝 Feel free to discuss with us privately!
