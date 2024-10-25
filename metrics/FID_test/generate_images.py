import torch
from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import argparse

import numpy as np
import random
import os
import copy
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Imagenette Generator',
        description='A script for generating specific image class of imagenette')
    parser.add_argument('--input_file', default='./coco_30k.csv')   
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default="./Models/stable-diffusion-v1-4/")
    parser.add_argument('--backdoor_model', default="backdoor_1")
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--origin', default=False, type=bool)
    
    return parser.parse_args()

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


def generate_with_seed(sd_pipeline, prompts, seed, guidance=7.5,output_path="./", num = '0', save_image=True):
    '''
    generate an image through diffusers 
    '''

    outputs = []
    generator = torch.Generator("cuda").manual_seed(seed)
    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt=prompt,generator=generator,guidance_scale=guidance,num_inference_steps=25)['images'][0]
        image_name = f"{output_path}/{num}.png"
        if save_image:
            image.save(image_name)
        # print("Saved to: ", image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model,safety_checker = None)
    # we disable the safety checker for test
    sd_pipeline = sd_pipeline.to(device)
    
    if not args.origin:
        epoch = args.epoch
        backdoor_name = args.backdoor_model
        path = f'../../results/{backdoor_name}/{epoch}'
        encoder = CLIPTextModel.from_pretrained(path)
        sd_pipeline.text_encoder = encoder.to(device)
        
        output_folder = f'./{backdoor_name}/{epoch}'
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                pass
    else:
        output_folder = f'./origin'
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                pass
    
    valid_set = pd.read_csv(args.input_file)
    for idx, raw_row in valid_set.iterrows():
        row = dict()
        for k,v in raw_row.items():
            row[k] = v
            
        prompt = row['prompt']
        seed = row['evaluation_seed']
        num = row['case_number']
        generate_with_seed(sd_pipeline, prompts=[prompt], seed=seed,output_path=output_folder,num=str(num))   
        