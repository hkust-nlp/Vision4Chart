import open_clip
# print(open_clip.list_pretrained())
import torch
from PIL import Image
import open_clip
import pandas as pd
import argparse
from tqdm import trange
import json
import wandb
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None

def find_max_dot_product(image_features, text1_features, text2_features):
    """
    Args:
    image_features (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the image features.
    text1_features (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the first set of text features.
    text2_features (torch.Tensor): A tensor of shape (batch_size, n, feature_dim) representing the second set of text features,
    or of shape (batch_size, feature_dim) if n=1.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the indices of the text features with the maximum dot product for each image feature.
    """
    batch_size, feature_dim = image_features.shape

    # Compute the dot product between image features and text1 features
    image_text1_dot = torch.bmm(image_features.view(batch_size, 1, feature_dim), text1_features.view(batch_size, feature_dim, 1)).squeeze(-1)

    # Check if text2_features has 2 dimensions and expand it if necessary
    if text2_features.dim() == 2:
        text2_features = text2_features.unsqueeze(1)

    # Compute the dot product between image features and each of the n text2 features
    image_text2_dot = torch.bmm(image_features.view(batch_size, 1, feature_dim), text2_features.transpose(1, 2))

    # Find the maximum dot product for each image feature across text1 and all text2 features
    text2_max_dot, text2_max_indices = torch.max(image_text2_dot, dim=-1)
    max_dot_product, max_indices = torch.max(torch.stack([image_text1_dot, text2_max_dot], dim=-1), dim=-1)
    final_indices = torch.where(max_indices == 0, max_indices, text2_max_indices + 1)
    return final_indices

def run_test(model, input_file_path, wandb_name, abs_path_prefix):
    wandb.init(project="clip_evaluation", name=wandb_name)
    
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

    model.eval()
    model = model.to(device)
    if '.csv' in input_file_path:
        print(f'--------loading csv file: {input_file_path}----------')
        df = pd.read_csv(input_file_path, sep=',')
        images_list = [os.path.join(abs_path_prefix, path) for path in df['img_path'].tolist()]
        captions = df['caption'].tolist()
        hard_captions = df['neg_caption'].tolist() 
        captions = [str(temp_cap) for temp_cap in captions]
        hard_captions = [str(temp_cap) for temp_cap in hard_captions]
    elif '.json' in input_file_path:
        data = load_json(input_file_path)
        if isinstance(data, dict):
            sorted_keys = sorted(data.keys())
            
            images_list = []
            captions = []
            hard_captions = []
            for k in sorted_keys:
                images_list.append(os.path.join(abs_path_prefix, data[k]['image']))
                captions.append(data[k]['statement'])
                hard_captions.append(data[k]['neg_statement'])
        elif isinstance(data, list):
            images_list = []
            captions = []
            hard_captions = []
            for temp_data in data:
                images_list.append(os.path.join(abs_path_prefix, temp_data['image']))
                captions.append(temp_data['statement'])
                hard_captions.append(temp_data['neg_statement'])
        
    correct_num = 0
    batch_size = 1
    index_record = []
    total_num = len(images_list)
    for i in trange(0, total_num, batch_size):
        img_paths = images_list[i : min(i+batch_size, total_num )]
        images = [Image.open(path) for path in img_paths]
        images = torch.stack([preprocess(img) for img in images]).to(device)
        positive_text = tokenizer(captions[i : min(i+batch_size, total_num )]).to(device)

        if isinstance(hard_captions[0],list):
            negative_text = tokenizer(hard_captions[i : min(i+batch_size, total_num )][0]).to(device)
        else:
            negative_text = tokenizer(hard_captions[i : min(i+batch_size, total_num )]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            positive_text_features = model.encode_text(positive_text)
            negative_text_features = model.encode_text(negative_text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            positive_text_features /= positive_text_features.norm(dim=-1, keepdim=True)
            negative_text_features /= negative_text_features.norm(dim=-1, keepdim=True)
            if isinstance(hard_captions[0],list):
                negative_text_features = negative_text_features.unsqueeze(0)
                
            max_indices = find_max_dot_product(image_features=image_features, text1_features=positive_text_features, text2_features=negative_text_features)
            index_record = index_record + max_indices.tolist()
            text1_count = (max_indices == 0).sum().item()
            correct_num += text1_count
            wandb.log({
            "batch": i // batch_size,
            "batch_accuracy": text1_count / batch_size
            })
    print(f"correct num: {correct_num}")
    print(f"total num: {total_num}")
    print(f"overall accuracy: {correct_num/total_num}")
    wandb.log({
        "correct_num": correct_num,
        "total_num": total_num,
        "overall_accuracy": correct_num / total_num
    })
    if args.save_result_path is not None:
        result_dict = {'input_file_path':args.input_file_path, 'model_file_path':args.model_file_path,
                       'correct num':correct_num, 'total num':total_num,  'overall accuracy':correct_num/total_num}
        with open(args.save_result_path, "w") as json_file:
            json.dump(result_dict, json_file)
        
        wandb.save(args.save_result_path)

    wandb.finish()
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_path', type=str, help='csv file path used to evaluate')
    parser.add_argument('--model_file_path', type=str, help='the path of clip model')
    parser.add_argument('--is_openai_baseline',  action="store_true",  help="Run openai baseline: clip-336-14") 
    parser.add_argument('--save_result_path', type=str, default=None, help='result file path')
    parser.add_argument('--wandb_name', type=str, default='evaluation_run', help='Name for the wandb run')
    parser.add_argument('--abs_path_prefix', type=str, default='./', help='Absolute path prefix for image paths')
    args = parser.parse_args()
    
    if args.is_openai_baseline:
        print("-----------loading the openai clip-336-14 model------------")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai', cache_dir='./data')
        
        run_test(model, args.input_file_path, args.wandb_name, args.abs_path_prefix)
    else:
        pretrained_path = args.model_file_path
        print(f"-----------loading clip model from {pretrained_path}-----------")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained=pretrained_path,
                                                                    force_quick_gelu=True)
        
        run_test(model, args.input_file_path, args.wandb_name, args.abs_path_prefix)
