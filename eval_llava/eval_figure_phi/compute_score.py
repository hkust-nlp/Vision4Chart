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

def rejudge_file(file_path, metric=exact_math):
    data = load_json(file_path=file_path)
    correct_num=0
    for k,temp_data in data.items():
        
        temp_answer = data[k]['answer']
        
        temp_response = data[k]['response']
        if ' <|end|>' in temp_response:
            temp_response = temp_response.split('<|end|>')[0]
        # if temp_response[-1] == "%":
        #     temp_response=temp_response[0:-1]
        temp_judge = metric(temp_response, temp_answer)
        data[k]['judge'] = temp_judge
        if temp_judge == True:
            correct_num+=1
    print(f"acc: {correct_num/len(data.keys())}")
    
def rejudge_file_chartX(file_path, metric=exact_math, is_extract=False):
    data = load_json(file_path=file_path)
    correct_num=0
    for k,temp_data in data.items():
        
        temp_answer = data[k]['answer']
        if is_extract:
            temp_answer = extract_number(temp_answer)
        temp_response = data[k]['response']
        if ' <|end|>' in temp_response:
            temp_response = temp_response.split('<|end|>')[0]
        if temp_response[-1] == "%":
            temp_response=temp_response[0:-1]
        temp_judge = metric(temp_response, temp_answer)
        data[k]['judge'] = temp_judge
        if temp_judge == True:
            correct_num+=1
    print(f"acc: {correct_num/len(data.keys())}")
    # save_json(data, file_path)
if __name__ == '__main__':
    model_notion = 'llava-v1.5-phi3-mini-no_hard_clip_chart_mix_reproduce-chart_mixed_v3_250k-full'
    PREV_PATH = '.'
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/plotqa/eval_results/{model_notion}_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/plotqa/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=relaxed_correctness)
    # rejudge_file(file_path=file_path2, metric=relaxed_correctness)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/figureqa/eval_results/{model_notion}_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/figureqa/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=exact_math)
    # rejudge_file(file_path=file_path2, metric=exact_math)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/dvqa/eval_results/{model_notion}_easy_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/dvqa/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_easy_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=exact_math)
    # rejudge_file(file_path=file_path2, metric=exact_math)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/dvqa/eval_results/{model_notion}_hard_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/dvqa/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_hard_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=exact_math)
    # rejudge_file(file_path=file_path2, metric=exact_math)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/chartbench/eval_results/{model_notion}_bin_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/chartbench/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_bin_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=exact_math)
    # rejudge_file(file_path=file_path2, metric=exact_math)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/chartbench/eval_results/{model_notion}_nqa_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/chartbench/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_nqa_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=relaxed_correctness)
    # rejudge_file(file_path=file_path2, metric=relaxed_correctness)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/chartqa/eval_results/{model_notion}_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/chartqa/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_responses_sampled.json'
    rejudge_file(file_path=file_path1, metric=relaxed_correctness)
    # rejudge_file(file_path=file_path2, metric=relaxed_correctness)
    print('------------')
    file_path1 = f'{PREV_PATH}/eval_llava/eval_data/chartX/eval_results/{model_notion}_responses_sampled.json'
    # file_path2 = 'eval_llava/eval_data/chartX/eval_results/llava-v1.5-phi3-mini-negclip_chart_mix_reproduce-chart_mixed_v3_250k-full_responses_sampled.json'
    rejudge_file_chartX(file_path=file_path1, metric=relaxed_correctness, is_extract=True)
    # rejudge_file_chartX(file_path=file_path2, metric=relaxed_correctness, is_extract=True)
    