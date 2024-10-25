```CUDA_VISIBLE_DEVICES=0,1 python backdoor_injection_main.py \
 -c './configs/backdoor_invisible/backdoor_1.yaml' \
 -l 1e-2 \
 -t './data/train/backdoor_1.txt'
 -p False
 ```

```CUDA_VISIBLE_DEVICES=0,1 python backdoor_injection_main.py \
 -c './configs/backdoor_invisible/backdoor_1.yaml' \
 -l 1e-2 \
 -t './data/train/backdoor_1.txt'
 -p True
 -pp './results/backdoor_1/'
 ```

evaluate FID
```
CUDA_VISIBLE_DEVICES=0 python ./metrics/FID_test/generate_images.py --backdoor_model backdoor_1 --epoch 599
CUDA_VISIBLE_DEVICES=0 python ./metrics/FID_test/fid_score.py --path1 ./coco_val.npz --path2 ./backdoor_1/599
```

evalute ASR
```
CUDA_VISIBLE_DEVICES=0 python ./metrics/ASR_test/generate_images_asr.py --backdoor_model backdoor_1 --epoch 599
```

evalute DSR
```
CUDA_VISIBLE_DEVICES=0 python ./metrics/DSR_test/generate_images_dsr.py --backdoor_model backdoor_1 --epoch 599
```