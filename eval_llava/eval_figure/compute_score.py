import json
from metric_utils import relaxed_correctness,exact_math
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
def compute_score(file_name, notion1=None):
    data = load_json(file_name)
    correct_num=0
    # print(len())
    if notion1 is None:
        for k,temp_data in data.items():
            if temp_data['judge'] == True:
                correct_num+=1
        return correct_num/len(data)
    else:
        notion1_num=0
        for k,temp_data in data.items():
            # print(temp_data)
            # break
            if 'source' in temp_data.keys():
                if temp_data['judge'] == True and temp_data['source']==notion1:
                    correct_num+=1
                if temp_data['source'] == notion1:
                    notion1_num+=1
            elif 'template' in temp_data.keys():
                if temp_data['judge'] == True and temp_data['template']==notion1:
                    correct_num+=1
                if temp_data['template'] == notion1:
                    notion1_num+=1
        return correct_num/notion1_num
    # return correct_num/75400
import re
def extract_number(input_str):
    numbers = re.findall(r'\b\d+\.?\d*', input_str)
    if numbers == []:
        return input_str
    return numbers[0]
def rejudge_file(file_path, metric=exact_math, is_extract=False):
    data = load_json(file_path=file_path)
    correct_num=0
    for k,temp_data in data.items():
        
        temp_answer = data[k]['answer']
        if is_extract:
            temp_answer = extract_number(temp_answer)
        temp_response = data[k]['response']
        if temp_response[-1] == "%":
            temp_response=temp_response[0:-1]
        temp_judge = metric(temp_response, temp_answer)
        data[k]['judge'] = temp_judge
        if temp_judge == True:
            correct_num+=1
    print(f"acc: {correct_num/len(data.keys())}")
    # save_json(data, file_path)
if __name__ == '__main__':
    model_notion = 'llava-v1.5-13b'
    file_path = f'eval_llava/eval_data/chartX/eval_results/{model_notion}_responses_sampled.json'
    # file_path = f'/cfs/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/eval_llava/eval_data/chartX/eval_results/{model_notion}_responses_sampled.json'
    rejudge_file(file_path=file_path, metric=relaxed_correctness, is_extract=True)
    # print(extract_number("$100 million"))
    # print(extract_number("25.00% hours"))
    # file_path = '/cfs/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/eval_llava/eval_data/chartX/eval_results/llava-v1.5-13b-chart_mixed_v3_250k_responses_sampled.json'
    # rejudge_file(file_path=file_path, metric=relaxed_correctness, is_extract=True)
    
    # file_path =  "/cfs/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/eval_llava/eval_data/chartX/eval_results/llava-v1.5-13b-chart_mixed_v3-full_responses_sampled.json"
    # rejudge_file(file_path=file_path, metric=relaxed_correctness, is_extract=True)
    
    # file_path =  "/cfs/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/eval_llava/eval_data/chartX/eval_results/llava-v1.5-13b-negclip_chart_mix_reproduce_self_5e-6_epoch3_chart_mixed_v3_250k_responses_sampled.json"
    # rejudge_file(file_path=file_path, metric=relaxed_correctness, is_extract=True)
    # file_path = 'eval_llava/eval_data/chartqa/eval_results/llava-v1.5-no_hard_clip_chartqa_responses.json'
    # rejudge_file(file_path=file_path)
    # file_path = 'eval_llava/eval_data/chartqa/eval_results/llava-v1.5-trained_chartqa_responses.json'
    # rejudge_file(file_path=file_path)
    # notion1 = 'structural'
    # file_name = 'eval_llava/eval_data/plotqa/eval_results/llava-v1.5-trained_chart_responses_sampled.json'
    # # rejudge_file(file_path=file_name)
    # acc = compute_score(file_name=file_name, notion1=notion1)
    # print(f"acc: {acc}")
    # file_name = 'eval_llava/eval_data/plotqa/eval_results/llava-v1.5-no-hard-clip_chart_responses_sampled.json'
    # # rejudge_file(file_path=file_name)
    # acc = compute_score(file_name=file_name, notion1=notion1)
    # print(f"acc: {acc}")
    # file_name = 'eval_llava/eval_data/plotqa/eval_results/llava-v1.5_neg_clip_chart_responses_sampled.json'
    # # rejudge_file(file_path=file_name)
    # acc = compute_score(file_name=file_name, notion1=notion1)
    # print(f"acc: {acc}")
    
    
    # file_name = 'eval_llava/eval_data/chartbench/eval_results/llava-v1.5-negclip_chartv2_responses.json'
    # acc = compute_score(file_name=file_name, notion1=None)
    # print(f"acc: {acc}")
    # acc = compute_score(file_name=file_name, notion1='human')
    # print(f"acc: {acc}")
    # rejudge_file('eval_llava/eval_data/plotqa/eval_results/llava-v1.5-trained_chart_responses_sampled.json')