import torch
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import random
from sklearn import decomposition
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

    
def pca(images,len_token):
    '''pca for an attention map group'''
    images_features = []
    i=0
    with torch.no_grad():
        for image in images:
            image_feature = image.flatten()
            images_features.append(image_feature)
        
    model = decomposition.PCA(n_components=20)
    model.fit(images_features)
    pca_matrix = model.fit_transform(images_features)
    
    return pca_matrix

def cov_m(features_tensor):
    '''Riemann logarithmic mapping'''
    features_tensor = torch.tensor(features_tensor)
    
    # compute the mean of the features
    mean_features = torch.mean(features_tensor, dim=0)
    
    # center the features
    centered_matrix = features_tensor - mean_features

    # compute the covariance matrix
    cov_matrix = torch.matmul(centered_matrix.t(), centered_matrix) / (centered_matrix.shape[0] - 1)

    # comptute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # compute the log of the eigenvalues
    log_eigenvalues = torch.log(eigenvalues)

    # make a diagonal matrix of the log of the eigenvalues
    log_eigenvalues_matrix = torch.diag_embed(log_eigenvalues)

    # compute the Riemannian logarithm of the centered matrix
    riemannian_log_mapping_matrix = eigenvectors @ log_eigenvalues_matrix @ eigenvectors.transpose(0, 1)
    
    return riemannian_log_mapping_matrix

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

    args = parse_args()
    
    # the LDA model
    lda = LinearDiscriminantAnalysis(n_components=1)
    
    with open('./train_model_reman_lda.pkl', 'rb') as f:
        lda = pickle.load(f)

    backdoor_path = args.backdoor_path

    for epoch in os.listdir(backdoor_path):
        epoch_path = os.path.join(backdoor_path,epoch)
        npy_path = os.path.join(epoch_path,'data.npy')
        load_dict = np.load(npy_path, allow_pickle=True).item()
        total_backdoor = 0
        for value in load_dict.values():
            images,length = value[0],value[1]
            input_data_backdoor = pca(images,length)
            input_data_backdoor = cov_m(input_data_backdoor)
            input_data_backdoor = np.expand_dims(input_data_backdoor, 0)
            input_data_backdoor = input_data_backdoor.reshape(input_data_backdoor.shape[0],-1)
            y_test_pred = lda.predict(input_data_backdoor)
            if y_test_pred:
                pass
            else:
                total_backdoor += 1

        with open(os.path.join(epoch_path,'backdoor_count_lda.txt'),'w',encoding='utf-8') as fout:
            fout.write(str(total_backdoor))

                
if __name__=='__main__':
    main()
    