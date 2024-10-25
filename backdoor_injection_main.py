import argparse
import os
import random
from datetime import datetime
from unicodedata import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config_parser import ConfigParser
from diffusers import StableDiffusionPipeline
from typing import List
import ptp_utils
import warnings
from tqdm import tqdm
from transformers import CLIPTextModel
import numpy as np
import abc

from diffusers import DPMSolverMultistepScheduler

# from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

LOW_RESOURCE = True
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

device_map = {
    'add_embedding': 0,
    'decoder': 0,
    'encoder': 0,
    'concept_embeds': 0,
    'concept_embeds_weights': 0,
    'special_care_embeds_weights': 0,
    'vision_model': 0,
    'visual_projection': 0,
    'conv_in': 0,
    'conv_out': 0,
    'post_quant_conv': 0,
    'special_care_embeds': 0,
    'text_model': 0,
    'conv_norm_out': 0,
    'quant_conv': 0,
    'time_embedding': 0,
    'text_projection': 0,
    'up_blocks': 1,
    'mid_block': 1,
    'down_blocks': 1,
}

ldm_stable = StableDiffusionPipeline.from_pretrained("./Models/stable-diffusion-v1-4/",device_map=device_map)
ldm_stable.scheduler = DPMSolverMultistepScheduler.from_config(ldm_stable.scheduler.config)
tokenizer = ldm_stable.tokenizer
LORA_USE = False


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
                h = attn.shape[0] # 16
                attn[h // 2:] = self.forward(attn[h // 2:].clone(), is_cross, place_in_unet)
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
    out_new = out.sum(0) / out.shape[0]
    return out_new

def preprocess(attention_store: AttentionStore, res: int, from_where: List[str], prompt: List[str], select: int = 0):
    tokens = tokenizer.encode(prompt[select])
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompt)
    attention_maps = attention_maps[:,:,:-1].clone().permute(2,0,1)
    
    return attention_maps,len(tokens)

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

    g_cpu = torch.Generator().manual_seed(int(seed))
    
    return g_cpu

def cov_m(features_tensor):
    '''Riemann logarithmic mapping'''
    features_tensor = features_tensor.flatten(start_dim=1)
    
    # compute the mean of the features
    mean_features = torch.mean(features_tensor, dim=0)
    
    # center the features
    centered_matrix = features_tensor.clone() - mean_features

    # compute the covariance matrix
    cov_matrix = torch.matmul(centered_matrix.t(), centered_matrix) / (centered_matrix.shape[0] - 1)

    return cov_matrix

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
        
class TextModifier:
    def __init__(self, text):
        self.text = text
        self.words_file = './text.txt'
        self.load_words()
    
    def load_words(self):
        """Load words from the file into the words list"""
        with open(self.words_file, 'r', encoding='utf-8') as file:
            self.words = [line.strip() for line in file if line.strip()]
    
    def insert_words(self):
        """Insert 1 to 5 random words from the file into the text at a random position"""
        if not self.words:
            print("Words file is empty or not loaded.")
            return
        
        num_words = random.randint(1, 5)  # Random number of words to insert
        words_to_insert = random.sample(self.words, num_words)
        position = random.randint(0, len(self.text.split()))
        words = self.text.split()
        words[position:position] = words_to_insert
        self.text = ' '.join(words)
    
    def delete_random_word(self):
        """Randomly delete a word from the text"""
        words = self.text.split()
        if words:
            word_to_delete = random.choice(words)
            words.remove(word_to_delete)
            self.text = ' '.join(words)
    
    def shuffle_text(self):
        """Shuffle the order of words in the text"""
        words = self.text.split()
        random.shuffle(words)
        self.text = ' '.join(words)
    
    def replace_word(self):
        """Randomly replace a word in the text with a random word from the file"""
        if not self.words:
            print("Words file is empty or not loaded.")
            return
        
        words = self.text.split()
        if words:
            index = random.randint(0, len(words) - 1)
            replacement_word = random.choice(self.words)
            words[index] = replacement_word
            self.text = ' '.join(words)
    
    def append_words(self):
        """Append 1 to 5 random words from the file to the end of the text"""
        if not self.words:
            print("Words file is empty or not loaded.")
            return
        
        num_words = random.randint(1, 5)  # Random number of words to append
        words_to_append = random.sample(self.words, min(num_words, len(self.words)))
        self.text += ' ' + ' '.join(words_to_append)
    
    def transform_text(self):
        """Randomly select an operation and apply it"""
        operations = [self.insert_words, self.delete_random_word, self.shuffle_text,
                      self.replace_word, self.append_words]
        operation = random.choice(operations)
        operation()

def main():
    g_cpu = set_seed(42)
    
    # define and parse arguments
    config, args = create_parser()
    torch.manual_seed(config.seed)

    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    dataset_name = args.train_dataset
    with open(dataset_name, 'r') as file:
        dataset = [line.strip() for line in file]
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # check for trigger overlappings
    print('######## Injected Backdoors ########')

    # load models
    if args.pretrained:
        path = args.pretrained_model_path
        encoder = CLIPTextModel.from_pretrained(path)
        ldm_stable.text_encoder = encoder.to(device)
        
        tokenizer = ldm_stable.tokenizer
        encoder_teacher = CLIPTextModel.from_pretrained(path).to(device) 
        encoder_student = CLIPTextModel.from_pretrained(path).to(device)
    else:
        tokenizer = ldm_stable.tokenizer
        encoder_teacher = CLIPTextModel.from_pretrained("./Models/stable-diffusion-v1-4/text_encoder/").to(device) 
        encoder_student = CLIPTextModel.from_pretrained("./Models/stable-diffusion-v1-4/text_encoder/").to(device)

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # fefine loss function
    loss_fkt = config.loss_fkt
    
    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    save_path = os.path.join(
        config.training['save_path'],
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)

    # training loop
    while (True):
        
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in config.backdoors:
                batch2 = []
                for sample in batch:
                    if backdoor['trigger'] not in sample:
                        modifier = TextModifier(sample)
                        modifier.transform_text()
                        clean_text = modifier.text
                        batch2.append(clean_text)

            batch_clean += batch2 
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]
            
        loss_benign = loss_fkt(embedding_student, embedding_teacher)

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:
            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                if config.injection['trigger_count']:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]
                else:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else: 
                        samples = [
                            sample for sample in batch
                        ]

                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [backdoor['target_prompt']],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]

            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]

                embedding_teacher_target = torch.repeat_interleave(
                    embedding_teacher_target,
                    len(embedding_student_backdoor),
                    dim=0)
            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target))

        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)

        loss_backdoor = torch.tensor(0.0).to(device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss
            
        # =========================== adapative attack ===========================
    
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=77, return_tensors="pt"
        )
        uncond_embeddings = encoder_student(uncond_input.input_ids.to(device))[0]
        
        
        mmd_loss = MMD_loss(kernel_type='rbf', kernel_mul=3.0, kernel_num=5)
        target_features = []
        backdoor_features = []

        # student
        for num in tqdm(range(len(batch_backdoor))):
            prompt = batch_backdoor[num]
            text = [prompt]
            controller = AttentionStore()
            text_input = text_input_backdoor.input_ids[num, :]
            embedding = embedding_student_backdoor[num,:,:].unsqueeze(0)
            ptp_utils.text2image_ldm_stable_v4(
                ldm_stable, text_input, embedding, uncond_embeddings, 1, controller,
                latent=None, num_inference_steps=NUM_DIFFUSION_STEPS,
                guidance_scale=GUIDANCE_SCALE, generator=g_cpu, 
                low_resource=LOW_RESOURCE, lora=LORA_USE)
            
            images_student_backdoor, _ = preprocess(controller, res=16, from_where=("up", "down"), prompt=text)
            input_student_backdoor = cov_m(images_student_backdoor)
            backdoor_features.append(input_student_backdoor)
                
        backdoor_distribution = torch.cat(backdoor_features, dim=0).to(device)
        
        # target
        with torch.no_grad():
            for prompt in tqdm(batch_clean):
                ldm_stable.text_encoder = encoder_teacher
                controller = AttentionStore()
                controller.reset()
                text = [prompt]
                controller = AttentionStore()
                ptp_utils.text2image_ldm_stable_v3(ldm_stable, text, controller, 
                                            latent=None, num_inference_steps=NUM_DIFFUSION_STEPS,
                                            guidance_scale=GUIDANCE_SCALE, generator=g_cpu, low_resource=LOW_RESOURCE,lora=LORA_USE)
                images_teacher_target,_ = preprocess(controller, res=16, from_where=("up", "down"), prompt=text)
                input_teacher_target = cov_m(images_teacher_target)
                target_features.append(input_teacher_target)
            
        target_distribution = torch.cat(target_features, dim=0).to(device)
        
        # compute mmd loss
        loss_reg = mmd_loss(backdoor_distribution, target_distribution)
    
        # =========================== adapative attack ===========================  
    
        loss = loss_benign + loss_backdoor + args.lambda_value*loss_reg
        
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()

        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        loss_reg = 0.1*loss_reg.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Reg Loss: {loss_reg:.4f} \t Total Loss: {loss_total:.4f}'
        )

        # update rtpt and lr scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()

        if (step+1) % 20 == 0:
            path = os.path.join(save_path,str(step))
            encoder_student.save_pretrained(f'{path}')
    
    path = os.path.join(save_path,'lambda.txt')
    with open(path,'w') as fin:
        fin.write(str(args.lambda_value)+'\n')


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default='./configs/backdoor_invisible/backdoor_1.yaml',
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('-p',
                        '--pretrained',
                        default=False,
                        type=bool,
                        dest="pretrained",
                        help='Whether to use pre-trained model (default: False)')
    parser.add_argument('-e',
                        '--epoch',
                        default=299,
                        type=int,
                        dest="epoch",
                        help='Epoch of pre-trained model (default: 299)')
    parser.add_argument('-b',
                        '--backdoor_model',
                        default='backdoor_1',
                        type=str,
                        dest="backdoor_model",
                        help='Backdoor model name (default: backdoor_1)')
    parser.add_argument('-l',
                        '--lambda_value',
                        default=1e-1,
                        type=float,
                        help='the coefficient of regularization term (default:1e-1)')
    parser.add_argument('-t',
                        '--train_dataset',
                        required=True,
                        type=str,
                        help='path of train dataset')
    parser.add_argument('-pp',
                        '--pretrained_model_path',
                        required=False,
                        default=None,
                        type=str,
                        help='path of pretrained model')
    
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args

if __name__ == '__main__':
    main()
    
