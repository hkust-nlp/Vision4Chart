# Evaluation script for question-answering task in ChartX
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Hancheng Ye, Renqiu Xia 
# All Rights Reserved 2024-2025.

import os
import json
import time
import logging
import argparse
import datetime
from tqdm import tqdm
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
import csv
from tqdm import tqdm
import time
import os
import random
import json
import openai
import fire

# from metric_utils import eval_gpt_acc

def eval_gpt_acc(question, answer_gt, answer_pred, key, patience=1000, sleep_time=0.2):
    # os.environ["https_proxy"] = "58.34.83.134:31280"
    # openai.api_base = 'https://api.openai.com/v1'
    openai.api_key = key

    examples = [
        {
            "query": "<question> What was the incremental increase in revenue from 2020 to 2021? <groundtruth answer> 5 million $ <answer> 20\n</s>",
            "answer": "False"
        },{
            "query": "<question> What percentage of government spending was allocated to infrastructure in 2020? <groundtruth answer> 10% <answer> 14-4=10\n</s>",
            "answer": "True"
        },{
            "query": "<question> What is the total production of Wind Energy in the four months from January to April 2021? <groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four months from January to April 2021 is 2450 MW.",
            "answer": "True"
        },{
            "query": "<question> What is the total of manufactured goods for UK and Germany combined? <groundtruth answer> 5 <answer> Five",
            "answer": "True"
        },
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # instruction
    prefix = f"""Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of predictions. The output should be only "True" or "False". Note that if the groundtruth answer is a numeric value with/without the unit, impose 5% error tolerance to the answer, e.g., the answer of 95 is marked as correct when groundtruth value is 100 million."""
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """
    
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    query = f"<question> {question} <groundtruth answer> {answer_gt} <answer> {answer_pred}"
    completion = None
    while patience > 0:
        patience -= 1
        try:
            completion = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo', 
                messages =[
                    {"role": "user",
                    "content": few_shot_prompt_template.format(
                        query=query
                        )
                    }
                ],
                api_key=key,
                temperature=0.
            )
            break
            # print(completion)
        except Exception as e:
            print(e)
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(promot) * 0.9)
                new_start = len(promot) - new_size
                promot = promot[new_start:]
                messages = [
                    {"role": "user", "content": promot},
                ]

            if sleep_time > 0:
                time.sleep(sleep_time)
                
    data_gen = completion.choices[0].message['content']
    if 'True' in data_gen:
        acc = 1
    if 'False' in data_gen: 
        acc = 0
    if 'True' not in data_gen and 'False' not in data_gen:
        acc = 0
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result_dir", required=True, help="Path to the inference result data")
    parser.add_argument("--your_openai_key", required=True, help="openai api key")
    parser.add_argument("--save_file_path", required=True, help="save file path")
    args = parser.parse_args()
    infer_result = args.infer_result_dir
    openai_key = args.your_openai_key
   

    len_sum = 0

    easy_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings',  'heatmap',  'box', 'candlestick', 'funnel', 'histogram', 'treemap']
    diff_types = ['rose', 'area_chart','3D-Bar','bubble','multi-axes', 'radar']
    single_class_chart = ['histogram', 'rose', 'rings', 'funnel', 'pie', 'treemap']
    chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'bar_chart_num', 'line_chart_num', 'rings', 'rose', 'area_chart', 'heatmap', '3D-Bar', 'box', 'bubble', 'candlestick', 'funnel', 'histogram', 'multi-axes', 'radar', 'treemap']

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"eval_result_qa_on_ChartX_{current_time}.log"
    os.makedirs(os.path.join("eval_result",'qa'), exist_ok=True)
    logging.basicConfig(filename=os.path.join("eval_result",'qa', log_file), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('infer_result:'+ infer_result)
    print("start processing")
    qa_score_set_total = []
    with open(infer_result) as json_file:  
        data = json.load(json_file)
    for c in tqdm(chart_types):
        qa_score_set = []

        for k,item in data.items():
            chart_type = item['image'].split('/')[8]
            # title = item["title_gt"]
            #imgname = item["imgname"]
            # csv_gt = item["csv_gt"]
            question = item["question"]
            answer_gt = item["answer"]
            answer_pred = item["response"]


            if chart_type == c:               
                
                qa_score = eval_gpt_acc(question, answer_gt, answer_pred, openai_key)
                logging.info(f"gt: {answer_gt}   pred: {answer_pred}  qa_score: {qa_score}") 
                # logging.info(f"the score: {qa_score}")
                item['gpt_score'] = qa_score
                qa_score_set.append(qa_score)
                qa_score_set_total.append(qa_score)      
        qa_score_type = sum(qa_score_set)/len(qa_score_set)
        logging.info('*************** Performance *****************')
        logging.info(c)
        logging.info('%.4f' % qa_score_type)

    with open(args.save_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    qa_score_total = sum(qa_score_set_total)/len(qa_score_set_total)
    logging.info('*************** Performance *****************')
    logging.info('average')
    logging.info('%.4f' % qa_score_total)