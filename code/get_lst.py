import openai
import json
import argparse
from tqdm import tqdm
from statistics import mean
import csv
import os
import re

gpt_4o_mini_ende_percentile_value = {'Addition': 99.75274, 'Inappropriate_for_context': 99.999885, 'Inconsistent_use': 99.997847, 'Register': 99.999516,
                                     'Awkward': 99.999981, 'Name_format': 100, 'Telephone_format': 100, 'Time_format': 99.997849, 'Omission': 99.979648,
                                     'Punctuation': 100, 'Spelling': 99.999766, 'Inconsistency': 99.996731, 'Character_encoding': 99.999938, 'Address_format': 100,
                                     'Currency_format': 100, 'Date_format': 99.984153, 'Untranslated_text': 99.907708, 'Mistranslation': 99.996052, 'Grammar': 99.999976}

qwen_max_ende_percentile_value = {'Addition': 100, 'Inappropriate_for_context': 100, 'Inconsistent_use': 100, 'Register': 100,
                                  'Awkward': 100, 'Name_format': 100, 'Telephone_format': 100, 'Time_format': 100, 'Omission': 100,
                                  'Punctuation': 100, 'Spelling': 100, 'Inconsistency': 100, 'Character_encoding': 100, 'Address_format': 100,
                                  'Currency_format': 100, 'Date_format': 100, 'Untranslated_text': 100, 'Mistranslation': 100, 'Grammar': 100}

qwen25_72b_ende_percentile_value = {'Addition': 100, 'Inappropriate_for_context': 100, 'Inconsistent_use': 100, 'Register': 100,
                                    'Awkward': 100, 'Name_format': 100, 'Telephone_format': 100, 'Time_format': 100, 'Omission': 100,
                                    'Punctuation': 100, 'Spelling': 100, 'Inconsistency': 100, 'Character_encoding': 100, 'Address_format': 100,
                                    'Currency_format': 100, 'Date_format': 100, 'Untranslated_text': 100, 'Mistranslation': 100, 'Grammar': 100}

gpt_4o_mini_zhen_percentile_value = {'Addition': 99.908895, 'Inappropriate_for_context': 99.999981, 'Inconsistent_use': 99.999945, 'Register': 100,
                                     'Awkward': 100, 'Name_format': 100, 'Telephone_format': 100, 'Time_format': 100, 'Omission': 99.984153,
                                     'Punctuation': 100, 'Spelling': 99.994164, 'Inconsistency': 99.999933, 'Character_encoding': 99.142243, 'Address_format': 99.997239,
                                     'Currency_format': 100, 'Date_format': 100, 'Untranslated_text': 100, 'Mistranslation': 100, 'Grammar': 100}

qwen_max_zhen_percentile_value = {'Addition': 100, 'Inappropriate_for_context': 100, 'Inconsistent_use': 100, 'Register': 100,
                                  'Awkward': 100, 'Name_format': 100, 'Telephone_format': 100, 'Time_format': 100, 'Omission': 100,
                                  'Punctuation': 100, 'Spelling': 100, 'Inconsistency': 100, 'Character_encoding': 100, 'Address_format': 100,
                                  'Currency_format': 100, 'Date_format': 100, 'Untranslated_text': 100, 'Mistranslation': 100, 'Grammar': 100}

qwen25_72b_zhen_percentile_value = {'Addition': 100, 'Inappropriate_for_context': 100, 'Inconsistent_use': 100, 'Register': 100,
                                    'Awkward': 100, 'Name_format': 100, 'Telephone_format': 100, 'Time_format': 100, 'Omission': 100,
                                    'Punctuation': 100, 'Spelling': 100, 'Inconsistency': 100, 'Character_encoding': 100, 'Address_format': 100,
                                    'Currency_format': 100, 'Date_format': 100, 'Untranslated_text': 100, 'Mistranslation': 100, 'Grammar': 100}


def read_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def save_json(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {path}.')
    return

def read_next_id(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)["next_id"]

def update_probs_lst(log_path, probs_lst, round, agent_type, model, lang):
    probs_dicts = read_json(log_path)

    if probs_lst == []:

        if lang == 'zhen':
            if model == 'gpt-4o-mini':
                threshold = gpt_4o_mini_zhen_percentile_value[agent_type]
            elif model == 'qwen-max':
                threshold = qwen_max_zhen_percentile_value[agent_type]
            elif model == 'qwen2.5-72b-instruct':
                threshold = qwen25_72b_zhen_percentile_value[agent_type]
        elif lang == 'ende':
            if model == 'gpt-4o-mini':
                threshold = gpt_4o_mini_ende_percentile_value[agent_type]
            elif model == 'qwen-max':
                threshold = qwen_max_ende_percentile_value[agent_type]
            elif model == 'qwen2.5-72b-instruct':
                threshold = qwen25_72b_ende_percentile_value[agent_type]
        else:
            raise ValueError('No corresponding threshold dictionary')

        for probs_dict in probs_dicts:
            if 'No' in probs_dict[2]['Error Exist']:
                probs_lst.append(False)
            else:
                if probs_dict[2]['probs'] <= threshold:
                    probs_lst.append(True)
                else:
                    probs_lst.append(False)
    else:
        for index, probs_dict in enumerate(probs_dicts):
            if probs_lst[index] == False:
                continue
            else:
                if 'I agree with' not in probs_dict[3][f'round{round - 1}']:
                    probs_lst[index] = True
                else:
                    probs_lst[index] = False
    return probs_lst

def get_error_exist_lst(log_path):
    error_exist_lst = []
    log_dicts = read_json(log_path)
    for log_dict in log_dicts:
        if 'Yes' in log_dict[0]['Error Exist']:
            error_exist_lst.append(('Yes', log_dict[0]['Severity']))
        else:
            error_exist_lst.append(('No', log_dict[0]['Severity']))
    return error_exist_lst
