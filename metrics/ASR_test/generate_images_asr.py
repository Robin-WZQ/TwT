import torch
from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import ptp_utils

import numpy as np
import random
import os
import copy
import pandas as pd
import PIL.Image as Image
from typing import Optional, Union, Tuple, List, Callable, Dict
import abc
import torch.nn.functional as nnf

import warnings
warnings.filterwarnings("ignore")

MAX_NUM_WORDS = 77
LOW_RESOURCE = False

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Imagenette Generator',
        description='A script for generating specific image class of imagenette')
    parser.add_argument('--input_file', default='./data/validation/poison_data_validation.txt')   
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default="./Models/stable-diffusion-v1-4/")
    parser.add_argument('--backdoor_model', default="2024-10-10_03-47-14")
    parser.add_argument('--epoch', default=299, type=int)
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
    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        # attn = attn.to("cpu")
        # self.step_store[key].append(attn)
        # attn = attn.to("cuda:0")
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt: List[str]):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompt), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

        
def preprocess(attention_store: AttentionStore, res: int, from_where: List[str], prompt: List[str], select: int = 0,tokenizer=None):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompt)
    images = []
    for i in range(1,77):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image))
        images.append(image[:,:,0])
    
    return images,len(tokens)

def generate_with_seed(sd_pipeline, prompts, seed, guidance=7.5,output_path="./", num = '0', save_image=True,tokenizer=None):
    '''
    generate an image through diffusers 
    '''
    generator = torch.Generator().manual_seed(seed)
    print(prompts[0])

    controller = AttentionStore()
    controller.reset()
    images, x_t = ptp_utils.text2image_ldm_stable_v2(sd_pipeline, prompts, controller, latent=None, num_inference_steps=50, guidance_scale=guidance, generator=generator, low_resource=False)
    
    attention_maps,length = preprocess(controller, res=16, from_where=("up", "down"), prompt=prompts,tokenizer=tokenizer)

    num_rows=1
    offset_ratio=0.02
    if type(images) is list:
        num_empCty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if save_image:
        path = f"{output_path}/{num}.png"
        pil_img.save(path .format(str(id)))
    
    return attention_maps,length

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model,safety_checker = None)
    # we disable the safety checker for test
    sd_pipeline = sd_pipeline.to(device)
    tokenizer = sd_pipeline.tokenizer
    controller = AttentionStore()

    if not args.origin:
        epoch = args.epoch
        backdoor_name = args.backdoor_model
        path = f'./results/{backdoor_name}/{epoch}'
        encoder = CLIPTextModel.from_pretrained(path)
        sd_pipeline.text_encoder = encoder.to(device)
        
        output_folder = f'./ASR/{backdoor_name}/{epoch}'
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                pass
    else:
        output_folder = f'./ASR/origin'
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                pass
    
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        prompts = [line.strip() for line in lines]

        pre_data = {}
        for i in range(len(prompts)):
            prompt = prompts[i]
            seed = 42
            num = i
            images,length = generate_with_seed(sd_pipeline, prompts=[prompt], seed=seed,output_path=output_folder,num=str(num),tokenizer=tokenizer)
            pre_data[i]=(images,length)
        
        np.save(output_folder + '/data.npy', pre_data)
        # the data.npy can be used to test detection success rate on FTT / LDA. 
        