import os
import re
import time
import argparse

from tqdm import tqdm

import sys

sys.path.append('../')
from utilities import *

# OpenAI
import openai


api_key = "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" #
headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
# load demo prompt
from ext_ans import demo_prompt


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response): #few
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, llm_engine, quick_extract=False):
    # print(f"quick_extract: {quick_extract} ")
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass
          
    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        # extraction = get_chat_response_new(full_prompt, headers, model=llm_engine, sleep_time=0.1)
        extraction = get_chat_response(promot=full_prompt, api_key=api_key, model=llm_engine)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return response


def extract_answer_quick(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        try:
            result = response.split('The answer is ')
            if result:
                #extraction = result.group(1)
                extraction = result[1]
                return extraction
        except:
            pass


    return response


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='./mathvista_outputs')
    parser.add_argument('--output_file', type=str, default='responses.json')
    parser.add_argument('--response_label', type=str, default='response',
                        help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices=['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613'])
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', default=False, help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', default=False, help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    args = parser.parse_args()
    print(args)
    # args
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != '':
        output_file = result_file.replace('.json', f'_{args.output_label}.json')
    else:
        output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    # full pids
    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print("Number of testing problems:", len(full_pids))

    # test pids
    if args.rerun:
        test_pids = full_pids
    else:
        test_pids = []
        for pid in full_pids:
            # print(pid)
            if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
                test_pids.append(pid)

    test_num = len(test_pids)
    print("Number of problems to run:", test_num)
    # print(test_pids)

    # tqdm, enumerate results
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]

        extraction = extract_answer(response, problem, args.llm_engine, args.quick_extract)
        results[pid]['extraction'] = extraction
        print(f"extraction: {extraction}")
        if i % args.save_every == 0 or i == test_num - 1:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
