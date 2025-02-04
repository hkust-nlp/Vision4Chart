from PIL import Image
import requests
from convert_vision import clip_convert_openclip, openclip_convert_clip
from transformers import CLIPProcessor, CLIPModel
import open_clip
import torch
from collections import OrderedDict
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def add_state_key_prefix(state_dict, prefix):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = prefix + key
        new_state_dict[new_key] =  value
    return new_state_dict

def get_clip_vision(model):
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device).eval()
    # print(model)
    model_dict = model.state_dict()
    # print(model_dict.keys())
    # print('------------------')
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    clip_vision_model = model.vision_model
    return clip_vision_model
    
    # return clip_vision_model_dict

def get_openclip_vision(model):
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='/ssddata/model_hub/junteng')
    # tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    openclip_vision_model = model.visual
    return openclip_vision_model
    
    # return openclip_vision_model_dict

def generate_clip(model):
    model.to(device)
    model.eval()
    print(model)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=["remote control", "two cats", "a cat", "there is no cats"], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        print(image_features[0,0:100])
        print(image_features.shape)
        print(text_features.shape)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(text_probs)
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    # print(probs)
def generate_openclip(model):
    model.to(device)
    model.eval()
    print(model)
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='./data')
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    # print(model.config)
    # model = model.to(device)
    # print(model)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
    # image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = tokenizer(["remote control", "two cats", "a cat", "there is no cats"]).to(device)
    # print(text.shape)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)

        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print(image_features[0,0:100])
        print(image_features.shape)
        print(text_features.shape)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(text_probs)
    
def check1():
    #check the correctness of function: clip_convert_openclip
    
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    generate_clip(model=model_clip)
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='/ssddata/model_hub/junteng')
    generate_openclip(model)
    
    clip_vision_model = get_clip_vision()
    clip_vision_model_dict = clip_vision_model.state_dict()
    # print(clip_vision_model_dict.keys())
    clip_vision_model_dict = add_state_key_prefix(state_dict=clip_vision_model_dict, prefix='vision_model.')
    # print(clip_vision_model_dict.keys())
    # print('----------------------')
    openclip_vision_model = get_openclip_vision()
    openclip_vision_model_dict = openclip_vision_model.state_dict()
    # print(openclip_vision_model_dict.keys())
    new_openclip_vision_model_dict = clip_convert_openclip(clip_vision_model_dict=clip_vision_model_dict, openclip_vision_model_dict=openclip_vision_model_dict)
    # print(new_openclip_vision_model_dict.keys())
    # create_shape = new_openclip_vision_model_dict['ln_post.bias'].shape
    # print(new_openclip_vision_model_dict['ln_post.bias'])
    # new_openclip_vision_model_dict['ln_post.bias'] = torch.zeros(create_shape)
    openclip_vision_model.load_state_dict(new_openclip_vision_model_dict, strict=False)
    
    model.visual = openclip_vision_model
    print("new model")
    generate_openclip(model)
    
def check2():
    #check the correctness of function: openclip_convert_clip
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    generate_clip(model=clip_model)
    print(clip_model.state_dict().keys)
    openclip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='/ssddata/model_hub/junteng')
    generate_openclip(openclip_model)
    
    clip_vision_model = get_clip_vision(clip_model)
    clip_vision_model_dict = clip_vision_model.state_dict()
    # print(clip_vision_model_dict.keys())
    clip_vision_model_dict = add_state_key_prefix(state_dict=clip_vision_model_dict, prefix='vision_model.')
    # print(clip_vision_model_dict.keys())
    # print('----------------------')
    openclip_vision_model = get_openclip_vision(openclip_model)
    openclip_vision_model_dict = openclip_vision_model.state_dict()
    # print(openclip_vision_model_dict.keys())
   
    new_clip_vision_model_dict = openclip_convert_clip(openclip_vision_model_dict=openclip_vision_model_dict,  clip_vision_model_dict=clip_vision_model_dict)
    # print(new_openclip_vision_model_dict.keys())

    print('------------------------')
    print(new_clip_vision_model_dict.keys())
    clip_model.load_state_dict(new_clip_vision_model_dict, strict=False)
    generate_clip(model = clip_model)
    # model.visual = openclip_vision_model
    # print("new model")
    # generate_openclip(model)

def conduct_convert(openclip_path, save_clip_path):
    #replace the vision part from openclip to clip
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="data/vision4math_clip_model/clip-vit-large-patch14-336")
    # generate_clip(model=clip_model)
    # print(clip_model.state_dict().keys)
    # openclip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='/ssddata/model_hub/junteng')
    # pretrained_path='/home/data/junteng/Vision4Math/train_clip/sum_logs/2024_07_15-21_16_53-model_ViT-L-14-336-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_3.pt'
    pretrained_path = openclip_path
    openclip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained=pretrained_path,
                                                                    force_quick_gelu=True)
    # generate_openclip(openclip_model)
    
    clip_vision_model = get_clip_vision(clip_model)
    clip_vision_model_dict = clip_vision_model.state_dict()
    clip_vision_model_dict = add_state_key_prefix(state_dict=clip_vision_model_dict, prefix='vision_model.')

    openclip_vision_model = get_openclip_vision(openclip_model)
    openclip_vision_model_dict = openclip_vision_model.state_dict()
    new_clip_vision_model_dict = openclip_convert_clip(openclip_vision_model_dict=openclip_vision_model_dict,  clip_vision_model_dict=clip_vision_model_dict)


    # print('------------------------')
    
    clip_model.load_state_dict(new_clip_vision_model_dict, strict=False)
    # generate_clip(model = clip_model)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="data/vision4math_clip_model/clip-vit-large-patch14-336")
    # torch.save(clip_model.state_dict(), '/ssddata/model_hub/junteng/vision4math_clip_model/trained_negclip_14_336/pytorch_model.bin')
    # torch.save(clip_model.state_dict(), save_clip_path)
    clip_model.save_pretrained(save_clip_path, safe_serialization=False)
    processor.save_pretrained(save_clip_path)
    print(f"save done in the path:{save_clip_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process save openclip into clip.')
    parser.add_argument('--openclip_path', type=str, help='Path to the open clip')
    parser.add_argument('--save_clip_path', type=str, help='Path to save the clip')

    args = parser.parse_args()
    # check2()
    # openclip_path='/project/deemreason/junteng/Vision4Math/train_clip/negative_logs/chartqa_v2/2024_08_30-17_35_06-model_ViT-L-14-336-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_10.pt'
    # save_clip_path='/project/deemreason/junteng/Vision4Math/data/vision4math_clip_model/trained_chartqav2_negclip_14_336/pytorch_model.bin'
    # openclip_path='/project/deemreason/junteng/Vision4Math/train_clip/negative_logs/plotqa_v2/2024_09_02-19_36_58-model_ViT-L-14-336-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_3.pt'
    # save_clip_path='/project/deemreason/junteng/Vision4Math/data/vision4math_clip_model/trained_plotqav2_negclip_14_336/pytorch_model.bin'
    # openclip_path='/project/deemreason/junteng/Vision4Math/train_clip/no_hard_negative_logs/plotqa_v2/2024_09_02-17_00_26-model_ViT-L-14-336-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_3.pt'
    # save_clip_path='/project/deemreason/junteng/Vision4Math/data/vision4math_clip_model/trained_plotqav2_no_hard_clip_14_336/pytorch_model.bin'
    openclip_path = args.openclip_path
    save_clip_path = args.save_clip_path
    conduct_convert(openclip_path=openclip_path, save_clip_path=save_clip_path)
    pass
    # openclip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='/ssddata/model_hub/junteng')
    # generate_openclip(openclip_model)

    # openclip_model.to(torch.float32)
    # torch.save(openclip_model.state_dict(), '/ssddata/model_hub/junteng/openclip-vit-14-336/openclip_model.pt')
    # openclip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336',  pretrained='openai',cache_dir='./')
    # generate_openclip(openclip_model)