import os
import io
import time
import argparse
from tqdm import tqdm
import sys
import wandb

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, evalmodel
from llava.utils import disable_torch_init
from utils import read_json, save_json
from metric_utils import exact_math, relaxed_correctness,relaxed_correctness_chartX
from prompt_utils import prompt_templates

import sys
import os
sys.path.insert(0, os.path.abspath("eval_llava/eval_figure/"))
import llava.model.builder
print(llava.model.builder.__file__)

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
    parser.add_argument('--abs_path_prefix', type=str, default='./', help='Absolute path prefix for image files')
    parser.add_argument('--conv_mode', type=str, default='v1', help='Conversation mode')
    args = parser.parse_args()
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
    image_list = [os.path.join(args.abs_path_prefix, temp_data['image']) for temp_data in data]
    
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
        
    model_path = args.model_path
    model_base = args.model_base

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path)
    )

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    
    if args.metric == 'em':
        metric_function = exact_math
    elif 'relaxed_cordrectness_chartX' in args.metric:
        metric_function = relaxed_correctness_chartX
    elif 'relax' in args.metric:
        metric_function = relaxed_correctness
    
    # tqdm, enumerate results
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

        if args.debug:
            print("--------------------------------------------------------------")

        try:
            args_llava = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": query,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ";",
                "temperature": args.temperature,
                "top_p": None,
                "num_beams": args.num_beam,
                "max_new_tokens": 512,
                "conv_mode": args.conv_mode,
                "num_return_sequences": args.num_return_sequences
            })()
            response = evalmodel(args_llava, model_name, tokenizer, model, image_processor, context_len)
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