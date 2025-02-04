import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from transformers import AutoProcessor, AutoModelForVision2Seq
import argparse
import sys
import os
import io
from tqdm import tqdm
import time
import wandb
import json
# from utils import read_json, save_json
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
from metric_utils import exact_math, relaxed_correctness
from prompt_utils import prompt_templates

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True

def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='./mathvista_data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='./mathvista_outputs')
    parser.add_argument('--output_file', type=str, default='responses.json')
    parser.add_argument('--summary_file', type=str, default='summary.json')
    # model
    parser.add_argument('--model_path', type=str, default='liuhaotian/llava-v1.5-13b', help='path of lora or full model')
    parser.add_argument('--model_base', type=str, default=None, help='liuhaotian/llava-v1.5-13b for lora, =None for full model')
    # query
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    # other settings
    parser.add_argument('--temperature', type=float, default=0., help='temperature when generating')
    parser.add_argument('--rerun', default=False, help='rerun answer extraction for all problems')
    parser.add_argument('--debug', default=False, help='debug mode')
    parser.add_argument('--prompt_template', default="", help='select the prompt template')
    parser.add_argument('--save_frequency',type=int, default=200, help='the frequency to save the response')
    parser.add_argument('--metric',type=str, default='em', help='the metric to evaluate')
    parser.add_argument('--wandb_name', type=str, default='llava_evaluation_run', help='Name for the wandb run')
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument('--num_beam', type=int, default=1, help='number of shot examples')
    args = parser.parse_args()
    DEVICE = "cuda"
    if args.debug:
        wandb.init(project="llava_evaluation_debug", name=args.wandb_name)
    else:
        wandb.init(project="llava_evaluation_debug", name=args.wandb_name)
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    
    selected_template = prompt_templates[args.prompt_template]
    
    question_list = [temp_data['question'] for temp_data in data]
    query_list=[selected_template.format(Question=temp_data['question']) for temp_data in data]
    answer_list = [temp_data['answer'] for temp_data in data]
    image_list = [temp_data['image'] for temp_data in data]
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # load results
    if os.path.exists(output_file):
        # print("\nResults already exist.")
        # print(f"Reading {output_file}...")
        # results = read_json(output_file)
        results = {}
    else:
        results = {}
        
    model = AutoModel.from_pretrained("data/idefics2-8b")
    model = model.to(device=DEVICE)

    processor = AutoProcessor.from_pretrained("data/idefics2-8b")
    model.eval()
    
    if args.metric == 'em':
        metric_function = exact_math
    elif 'relax' in args.metric:
        metric_function = relaxed_correctness
    
    
    correct_count = 0
    total_count = 0
    for pid, _ in enumerate(tqdm(query_list)):
        ## for debug test
        # if pid > 400:
        #     break
        ## for debug test
        
        problem = data[pid]
        question = question_list[pid]
        query = query_list[pid]
        image = image_list[pid]
        answer = answer_list[pid]
        image_path = image
        
        image1 = Image.open(image_path).convert('RGB')
        question = query
        msgs =[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]
            }
        ]
        try:
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image1], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            response = generated_texts[0].split('\nAssistant:')[-1].strip()
            results[str(pid)] = problem
            results[str(pid)]['query'] = query
            results[str(pid)]['response'] = response
            if isinstance(response, list):
                is_correct_list = []
                for temp_response in response:
                    is_correct = metric_function(prediction=temp_response, target=answer)
                    is_correct_list.append(is_correct)
                results[str(pid)]['judge'] = is_correct_list
                # print(is_correct_list)
                if True in is_correct_list:
                    correct_count += 1
            else:
                is_correct = metric_function(prediction=response, target=answer)
                results[str(pid)]['judge'] = is_correct
                if is_correct:
                    correct_count += 1
            total_count += 1
            
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
            
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[str(pid)]['error'] = str(e)
        
        if pid % args.save_frequency == 0:
            try:
                print(f"Saving results to {output_file}...")
                save_json(results, output_file)
                print(f"Results saved.")
            except Exception as e:
                print(e)
                print(f"Error in saving {output_file}")
                
            # Save results to wandb after each iteration
            wandb.log({"results": results})
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    summary = {
        "accuracy": accuracy,
        "model_path": args.model_path,
        "input_file": args.input_file,
        "metric": args.metric,
        "total_samples": total_count,
        "correct_samples": correct_count
    }
    
    summary_file = os.path.join(args.output_dir, args.summary_file)
    try:
        print(f"Saving summary to {summary_file}...")
        save_json(summary, summary_file)
        print(f"Summary saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {summary_file}")
    
    try:
        print(f"Saving final results to {output_file}...")
        save_json(results, output_file)
        print(f"Final results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")
    
    wandb.log(summary)
    wandb.finish()
