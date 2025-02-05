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

words_list= ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
def words_to_number(word):
    word_dict = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20
    }
    return str(word_dict.get(word.lower()))



def exact_math(prediction, target) -> bool:
    def preprocess_string(text: str):
        # Convert to lowercase and remove punctuation
        return text.lower().strip('.,!? ')

    prediction_processed = preprocess_string(prediction)
    target_processed = preprocess_string(target)
    if prediction_processed in words_list:
        prediction_processed = words_to_number(prediction_processed)
    if target_processed in words_list:
        target_processed = words_to_number(target_processed)
    return prediction_processed == target_processed

def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """
    target = str(target)
    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    def preprocess_string(text: str):
        # Convert to lowercase and remove punctuation
        return text.lower().strip('.,!? ')

    prediction_processed = preprocess_string(prediction)
    target_processed = preprocess_string(target)
    # print(prediction_processed)
    prediction_float = _to_float(prediction_processed)
    target_float = _to_float(target_processed)
    # print(prediction_float)
    # print(target_float)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        # print(relative_change)
        return relative_change <= max_relative_change
    elif prediction_float is not None and target_float is not None:
        return prediction_float == target_float
    else:
        return prediction_processed == target_processed

import re
def extract_number(input_str):
    numbers = re.findall(r'\b\d+\.?\d*', input_str)
    if numbers == []:
        return input_str
    return numbers[0]

def relaxed_correctness_chartX(prediction, target) -> bool:
    temp_answer = extract_number(target)
    temp_response = prediction
    if temp_response[-1] == "%":
        temp_response=temp_response[0:-1]
    temp_judge = relaxed_correctness(temp_response, temp_answer)
    return temp_judge

if __name__ == '__main__':
    # print(exact_math("8.2.", "8.2"))
    print(relaxed_correctness('4.55e+10.', '4.87e+10.'))
