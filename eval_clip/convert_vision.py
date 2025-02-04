import torch
from transformers import AutoConfig
# from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
import json
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import open_clip
from collections import OrderedDict


def openclip_convert_clip(openclip_vision_model_dict, clip_vision_model_dict):
    """
    Args:
        openclip_vision_model_dict (OrderedDict): state_dict from OpenCLIP VisionTransformer
        clip_vision_model_dict (OrderedDict): state_dict for CLIP CLIPVisionTransformer
    """
    new_state_dict = OrderedDict()
    for key, value in openclip_vision_model_dict.items():
        new_key = key
        if 'attn.in_proj_' in key:
            new_key = new_key.replace('transformer.resblocks', 'vision_model.encoder.layers')
            q_proj_value, k_proj_value, v_proj_value = value.chunk(3, dim=0)
            new_key = new_key.replace('attn.in_proj_', 'self_attn.q_proj.')
            new_state_dict[new_key] = q_proj_value
            new_key = new_key.replace('q_proj', 'k_proj')
            new_state_dict[new_key] = k_proj_value
            new_key = new_key.replace('k_proj', 'v_proj')
            new_state_dict[new_key] = v_proj_value
            # print(f'new key: {new_key}')
            continue
        new_key = new_key.replace('class_embedding', 'vision_model.embeddings.class_embedding')
        new_key = new_key.replace('positional_embedding', 'vision_model.embeddings.position_embedding.weight')
        new_key = new_key.replace('conv1.weight', 'vision_model.embeddings.patch_embedding.weight')
        new_key = new_key.replace('ln_pre.weight', 'vision_model.pre_layrnorm.weight')
        new_key = new_key.replace('ln_pre.bias', 'vision_model.pre_layrnorm.bias')
        new_key = new_key.replace('transformer.resblocks', 'vision_model.encoder.layers')

        new_key = new_key.replace('attn.out_proj.weight', 'self_attn.out_proj.weight')
        new_key = new_key.replace('attn.out_proj.bias', 'self_attn.out_proj.bias')
        new_key = new_key.replace('ln_1.weight', 'layer_norm1.weight')
        new_key = new_key.replace('ln_1.bias', 'layer_norm1.bias')
        new_key = new_key.replace('ln_2.weight', 'layer_norm2.weight')
        new_key = new_key.replace('ln_2.bias', 'layer_norm2.bias')
        new_key = new_key.replace('mlp.c_fc.weight', 'mlp.fc1.weight')
        new_key = new_key.replace('mlp.c_fc.bias', 'mlp.fc1.bias')
        new_key = new_key.replace('mlp.c_proj.weight', 'mlp.fc2.weight')
        new_key = new_key.replace('mlp.c_proj.bias', 'mlp.fc2.bias')
        new_key = new_key.replace('ln_post.weight', 'vision_model.post_layernorm.weight')
        new_key = new_key.replace('ln_post.bias', 'vision_model.post_layernorm.bias')

        new_state_dict[new_key] = value

    print(f"Number of converted keys: {len(new_state_dict)}")
    
    # Check for missing keys in CLIP state_dict
    for clip_key in clip_vision_model_dict.keys():
        if clip_key not in new_state_dict:
            print(f"Missing key in converted state_dict: {clip_key}")

    return new_state_dict

def clip_convert_openclip(clip_vision_model_dict, openclip_vision_model_dict):
    """_summary_
    Args:
        clip_vision_model_dict (_type_): the class is CLIPVisionTransformer from transformers
        openclip_vision_model_dict (_type_): the class is VisionTransformer from OpenClip
    """
    new_state_dict = OrderedDict()
    for key, value in clip_vision_model_dict.items():
        new_key = key
        if 'self_attn.k_proj' in key or 'self_attn.v_proj' in key:
            continue
        new_key = new_key.replace('vision_model.embeddings.class_embedding', 'class_embedding')
        new_key = new_key.replace('vision_model.embeddings.patch_embedding', 'conv1')
        new_key = new_key.replace('vision_model.embeddings.position_embedding.weight', 'positional_embedding')
        new_key = new_key.replace('vision_model.pre_layrnorm', 'ln_pre')
        new_key = new_key.replace('vision_model.encoder.layers', 'transformer.resblocks')

        new_key = new_key.replace('self_attn.out_proj', 'attn.out_proj')
        new_key = new_key.replace('layer_norm1', 'ln_1')
        new_key = new_key.replace('layer_norm2', 'ln_2')
        new_key = new_key.replace('mlp.fc1', 'mlp.c_fc')
        new_key = new_key.replace('mlp.fc2', 'mlp.c_proj')
        new_key = new_key.replace('vision_model.post_layernorm', 'ln_post')

        if 'self_attn.q_proj.' in key:
            new_key = new_key.replace('self_attn.q_proj.', 'attn.in_proj_')
            new_state_dict[new_key] = torch.cat((value, clip_vision_model_dict[key.replace('q_proj', 'k_proj')], clip_vision_model_dict[key.replace('q_proj', 'v_proj')]), dim=0)
            if new_key not in openclip_vision_model_dict.keys()  :
                print(f"no this key in openclip: {new_key}")
            elif new_state_dict[new_key].shape != openclip_vision_model_dict[new_key].shape:
                print(f"wrong shape of {new_key}: in openclip shape is {openclip_vision_model_dict[new_key].shape}, while the current shape is {new_state_dict[new_key].shape}")
            continue

        new_state_dict[new_key] = value
        if new_key not in openclip_vision_model_dict.keys()  :
            print(f"no this key in openclip: {new_key}")
        elif new_state_dict[new_key].shape != openclip_vision_model_dict[new_key].shape:
            print(f"wrong shape of {new_key}: in openclip shape is {openclip_vision_model_dict[new_key].shape}, while the current shape is {new_state_dict[new_key].shape}")
        
    print(f"the number of converted keys: {len(new_state_dict.keys())}")
    assert len(new_state_dict.keys())==295
    # openclip_vision_model_dict.load_state_dict(new_state_dict, strict=False)
    for temp_key in openclip_vision_model_dict.keys():
        if temp_key not in new_state_dict.keys():
            print(f"the key: {temp_key} are not updated!")
    return new_state_dict

    # 保存新的模型权重
    # torch.save(clip_model.state_dict(), 'path_to_save/clip_model_with_llava_weights.pth')
if __name__ == '__main__':
    vision_config_path = "/ssddata/model_hub/junteng/llava_llama3_vision/vision_module_config.json"
    vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336")
    vision_module = CLIPVisionModel(vision_config)
    vision_weights_path = "/ssddata/model_hub/junteng/llava_llama3_vision/vision_module_weights.pth"
    vision_module.load_state_dict(torch.load(vision_weights_path))
    # print(vision_module.state_dict().keys())
    llava_vision_module_dict = vision_module.state_dict()
    print(f"llava_vision: clip vision number: {len(llava_vision_module_dict)}")
    print(llava_vision_module_dict.keys())
    print("---------")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai',cache_dir='/ssddata/model_hub/junteng')
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_vision_moudle = model.visual
    # print(clip_vision_moudle)
    clip_vision_moudule_dict = clip_vision_moudle.state_dict()
    print(f"open clip vision number: {len(clip_vision_moudule_dict)}")
    print(clip_vision_moudule_dict.keys())
    # print(model.config)
    # model = model.to(device)
    # print(model.visual.state_dict().keys())
    # print(llava_vision_module_dict['vision_model.encoder.layers.0.self_attn.v_proj.weight'].shape)
    # print(clip_vision_moudule_dict['transformer.resblocks.1.attn.in_proj_weight'].shape)

    # clip_convert_openclip(llava_vision_module_dict, clip_vision_moudule_dict)
    openclip_convert_clip(clip_vision_moudule_dict, llava_vision_module_dict)