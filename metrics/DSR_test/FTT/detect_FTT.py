import torch
import numpy as np
import os
import random
import warnings
import argparse

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def find_max(images,len_tokens):
    max_num = images[0]/255
    for image in images[1:len_tokens]:
        max_num = np.add(max_num,image/255)

    high_atm = max_num / len_tokens
    return high_atm,images

def compute_ssim(high_atm,images,length):
    values = []
    for i in range(length-1):
        image = images[i]/255
        high_atm = high_atm
        # value = sum(map(sum,(high_atm - image)**2))
        value = np.linalg.norm(high_atm - image, 'fro')
        values.append(value)
        # print(np.mean(value))
    re = np.mean(values)
    return re

def compute(trigger,prompt_file_path,result_file_path):
    with open(prompt_file_path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        
    with open(result_file_path,'r',encoding='utf-8') as fin2:
        results = fin2.readlines()
        
    assert len(lines)==len(results)

    tp,tn,fp,fn = 0,0,0,0
        
    for idx in range(len(lines)):
        label = 0
        if trigger in lines[idx]:
            label = 1
        predicted = 0
        if results[idx].strip() == 'backdoor':
            predicted = 1
                 
        tp += (predicted == 1) & (label == 1)
        tn += (predicted == 0) & (label == 0)
        fp += (predicted == 1) & (label == 0)
        fn += (predicted == 0) & (label == 1)
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision*100, recall*100, f1_score*100


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the backdoor detection performance'
    )
    parser.add_argument('--backdoor_path', default='./ASR/backdoor_1')   
    
    return parser.parse_args()

def main():
    set_seed(42)  

    threshold = 2.5
    args = parse_args()
        
    backdoor_path = args.backdoor_path
    for epoch in os.listdir(backdoor_path):
        epoch_path = os.path.join(backdoor_path,epoch)
        npy_path = os.path.join(epoch_path,'data.npy')
        load_dict = np.load(npy_path, allow_pickle=True).item()
        total_backdoor = 0
        for value in load_dict.values():
            images,length = value[0],value[1]
            high_atm,images = find_max(images,length)
            y = round(compute_ssim(high_atm,images,length),3)
            print(y)
            if y < threshold:
                total_backdoor += 1
        with open(os.path.join(epoch_path,'backdoor_count.txt'),'w',encoding='utf-8') as fout:
            fout.write(str(total_backdoor))

                
if __name__=='__main__':
    main()
    