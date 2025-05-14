import openai
import json
import argparse
from tqdm import tqdm
from statistics import mean
import csv
import os
import re
import numpy as np


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


def get_logprobs(response):
    logprobs_str = str(response["logprobs"])
    logprobs = re.findall(r'logprob=([-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|[-+]?\d+)', logprobs_str)
    logprobs = [float(logprob) for logprob in logprobs]
    return logprobs

def compute_prob(logprobs):
    logprob_sum = sum(logprobs)
    probability = np.exp(logprob_sum)
    return probability

def record_SE(log_path, response_path):
    responses = read_json(response_path)
    states = []
    for res in responses:
        try:
            id = res["id"]
            error_exist = res["content"].split('Error Exist: ')[1].split('\n')[0]
            severity_match = re.search(r"Error Severity:\s*['\"]?(Major|Neutral|Minor)['\"]?", res["content"],re.IGNORECASE)
            explanation = res["content"].split('Explanation: ')[1]
            state = [
                {
                    "stage": 'SE',
                    "id": id,
                    "Error Exist": error_exist,
                    "Severity": severity_match.group(1),
                    "Explanation": explanation
                },
            ]
            states.append(state)
        except Exception as e:
            print(e)
            print(id)
    save_json(states, log_path)

def record_SRc(log_path, response_path, error_exist_lst):
    responses = read_json(response_path)
    data = read_json(log_path)
    count = 0
    for item, error_exist in zip(data, error_exist_lst):
        if 'Yes' in error_exist[0]:
            try:
                state = {"stage": 'SRc', "Corrected Translation": responses[count]["content"].split('Translation: ')[1]}
            except:
                print(count)
        else:
            state = {"stage": 'SRc', "Corrected Translation": 'None'}
        item.append(state)
        count += 1
    save_json(data, log_path)

def record_SRv(log_path, response_path, error_exist_lst):
    responses = read_json(response_path)
    data = read_json(log_path)
    count = 0
    for item in data:
        severity = item[0]['Severity']
        error_exist = item[0]['Error Exist']
        if 'Yes' in error_exist:
            if 'A' in  responses[count]["content"]:
                error_exist = 'Yes'
            elif 'B' in responses[count]["content"]:
                error_exist = 'No'
                severity = 'Neutral'

            logprobs_lst = get_logprobs(responses[count])
            prob = compute_prob(logprobs_lst)

            state = {"stage": 'SRv', "Error Exist": error_exist, "Severity": severity, "probs": prob * 100}
        else:
            state = {"stage": 'SRv', "Error Exist": error_exist, "Severity": severity, "probs": 0.0}
        item.append(state)
        count += 1
    save_json(data, log_path)

def record_CD(log_path, response_path, round):
    responses = read_json(response_path)
    data = read_json(log_path)
    count = 0
    for item in data:
        state = {"stage": 'CD'}
        if round == 0 and len(item) == 3:
            state[f"round{round}"] = responses[count]["content"]
            item.append(state)
        elif round > 0:
            item[3][f"round{round}"] = responses[count]["content"]
        count += 1
    save_json(data, log_path)

    if round == 3:
        data = read_json(log_path)
        for item in data:
                error_exist, severity = get_stage4_final_result(item)
                if 'No' in error_exist:
                    error_exist = 'No'
                    severity = 'Neutral'
                else:
                    error_exist = 'Yes'
                    if 'Major' in severity:
                        severity = 'Major'
                    elif 'Minor' in severity:
                        severity = 'Minor'
                    else:
                        severity = 'Neutral'
                item[3]['Error Exist'] = error_exist
                item[3]['Severity'] = severity
        save_json(data, log_path)

def get_stage4_final_result(stage_dict):
    if stage_dict[3]['round0'] == 'None' or stage_dict[3]['round1'] == 'None':
        return stage_dict[2]['Error Exist'], stage_dict[2]['Severity']

    if stage_dict[3]['round2'] == 'None':
        error_exist_pattern = r'Error Exist:\s*["\']?([\w\s.]+)["\']?'
        error_severity_pattern = r'Error Severity:\s*["\']?([\w\s.]+)["\']?'
        major_agent_error_exist_match = re.search(error_exist_pattern, stage_dict[3]['round0'])
        major_agent_error_severity_match = re.search(error_severity_pattern, stage_dict[3]['round0'])
        if major_agent_error_exist_match:
            error_exist = major_agent_error_exist_match.group(1)
            if major_agent_error_severity_match:
                error_severity = major_agent_error_severity_match.group(1)
            else:
                error_severity = 'Neutral' if 'No' in error_exist else 'Minor'
            return error_exist, error_severity
        else:
            stage_dict[2]['Error Exist'], stage_dict[2]['Severity']

    if 'I agree' in stage_dict[3]['round3']:
        error_exist_pattern = r'Error Exist:\s*(\w+)'
        error_severity_pattern = r'Error Severity:\s*(\w+)'
        major_agent_error_exist_match = re.search(error_exist_pattern, stage_dict[3]['round0'])
        major_agent_error_severity_match = re.search(error_severity_pattern, stage_dict[3]['round0'])
        if major_agent_error_exist_match and major_agent_error_severity_match:
            error_exist = major_agent_error_exist_match.group(1)
            error_severity = major_agent_error_severity_match.group(1)
            return error_exist, error_severity
        else:
            return stage_dict[2]['Error Exist'], stage_dict[2]['Severity']
    else:
        return stage_dict[2]['Error Exist'], stage_dict[2]['Severity']

def record_final(log_path):
    data = read_json(log_path)
    for item in data:
        if 'Yes' in item[3]['Error Exist']:
            error_exist = 'Yes'
            if 'Minor' in item[3]['Severity']:
                severity = 'Minor'
            elif 'Major' in item[3]['Severity']:
                severity = 'Major'
            else:
                severity = 'Neutral'
        else:
            error_exist = 'No'
            severity = 'Neutral'
        stage = {"stage": 'Final', "Error Exist": error_exist, 'Severity': severity}
        item.append(stage)
    save_json(data, log_path)


