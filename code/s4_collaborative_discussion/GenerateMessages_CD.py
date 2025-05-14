import json
import os
import argparse


def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError as e:
        print(f"Decode error：{e}")
        print(f"Error location：{e.start} to {e.end}")

def save_json(data, path):
    # Create the folder if non-existence
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {path}.')
    return

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

class MultiAgentPrompting_CD:
    def __init__(self, dataset, lang_pair, translation_system, agent_type, dataset_prefix, agent_type_prefix, probs_lst, messages_path, process_log_path):
        '''
        dataset: the different dataset of WMT General MT shared task, such as wmt20, wmt22
        lang_pair: the language pari to be evaluated, such as zhen, deen
        translation_system: the system that provide the translation which will be evaluated by ours framework, such as HuaweiTSC
        agnet_type: the agents responsible for different types of MQM errors, such as accuracy, fluency
        dataset_prefix: the path prefix of the used dataset
        agent_type_prefix: the path prefix of the used prompt which includes the definition of agent types
        probs_lst: the list about whether the discussion should go on
        messages_path: the path of the messages to be saved
        process_log_path: The processed log path
        '''
        self.lang_pair = lang_pair
        self.dataset = dataset
        self.translation_system = translation_system
        self.probs_lst = probs_lst
        self.messages_path = messages_path
        self.process_log_path = process_log_path

        self.tier2_agent_type = agent_type
        self.tier1_agent_type = self.get_tier1_agent_type()

        self.dataset_file_path = os.path.join(dataset_prefix, dataset, lang_pair, translation_system)
        self.agent_type_file_path = os.path.join(agent_type_prefix, lang_pair, f'description_CD.json')

    def get_tier1_agent_type(self):
        if self.tier2_agent_type == 'Addition' or self.tier2_agent_type == 'Omission' or self.tier2_agent_type == 'Mistranslation' or self.tier2_agent_type == 'Untranslated_text':
            return 'Accuracy'
        elif self.tier2_agent_type == 'Inappropriate_for_context' or self.tier2_agent_type == 'Inconsistent_use':
            return 'Terminology'
        elif self.tier2_agent_type == 'Punctuation' or self.tier2_agent_type == 'Spelling' or self.tier2_agent_type == 'Grammar' or self.tier2_agent_type == 'Register' or self.tier2_agent_type == 'Inconsistency' or self.tier2_agent_type == 'Character_encoding':
            return 'Fluency'
        elif self.tier2_agent_type == 'Awkward':
            return 'Style'
        elif self.tier2_agent_type == 'Address_format' or self.tier2_agent_type == 'Currency_format' or self.tier2_agent_type == 'Date_format' or self.tier2_agent_type == 'Name_format' or self.tier2_agent_type == 'Telephone_format' or self.tier2_agent_type == 'Time_format':
            return 'Locale_conventions'


    def generate_prompt_tier1_agent_round1(self):
        prompt_description = read_json(self.agent_type_file_path)
        for line in prompt_description:
            if line['description'] == 'tier1_agent_system_prompt':
                tier1_agent_system_prompt = line['content']
            if line['description'] == 'error_severity':
                error_severity = line['content']
            if line['description'] == f'{self.tier1_agent_type}':
                tier1_agent_type_description = line['content']
            if line['description'] == f'{self.tier2_agent_type}':
                tier2_agent_type_description = line['content']
            if line['description'] == f'tier1_agent_round1':
                tier1_agent_round1 = line['content']
        with open(os.path.join(self.dataset_file_path, 'source.txt'), 'r', encoding='utf-8') as srcs, \
            open(os.path.join(self.dataset_file_path, 'hyp.txt'), 'r', encoding='utf-8') as hyps:
            output_file_path = self.messages_path
            process_log_lines = read_json(self.process_log_path)
            messages_lst = []
            count = 0
            for (src, hyp, process) in zip(srcs, hyps, process_log_lines):
                if self.probs_lst[count] == True:
                    prompt = str(tier1_agent_type_description) + '\n' + str(error_severity) + str(tier1_agent_round1)
                    prompt = prompt + '\nThe subcategory evaluator\'s evaluation:'  + '\nSource: ' + src + 'Translation: ' + hyp
                    prompt += 'Error Type: ' + self.tier2_agent_type + '\nError Exist: ' + process[2]["Error Exist"] + '\nSeverity: ' + process[2]["Severity"]
                    message_lst = [
                        {"role": "system", "content": tier1_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    prompt = 'None'
                    message_lst = [
                        {"role": "system", "content": tier1_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                count += 1
                messages_lst.append(message_lst)
            save_json(messages_lst, output_file_path)

    def generate_prompt_tier2_agent_round1(self):
        prompt_description = read_json(self.agent_type_file_path)
        for line in prompt_description:
            if line['description'] == 'tier2_agent_system_prompt':
                tier2_agent_system_prompt = line['content']
            if line['description'] == 'error_severity':
                error_severity = line['content']
            if line['description'] == f'{self.tier1_agent_type}':
                tier1_agent_type_description = line['content']
            if line['description'] == f'{self.tier2_agent_type}':
                tier2_agent_type_description = line['content']
            if line['description'] == f'tier2_agent_round1':
                tier2_agent_round1 = line['content']
        with open(os.path.join(self.dataset_file_path, 'source.txt'), 'r', encoding='utf-8') as srcs, \
            open(os.path.join(self.dataset_file_path, 'hyp.txt'), 'r', encoding='utf-8') as hyps:
            output_file_path = self.messages_path
            process_log_lines = read_json(self.process_log_path)
            messages_lst = []
            count = 0
            for (src, hyp, process) in zip(srcs, hyps, process_log_lines):
                if self.probs_lst[count] == True:
                    prompt = f'You are responsible for {self.tier2_agent_type}. The definition of {self.tier2_agent_type} is: ' + str(tier2_agent_type_description) + str(error_severity) + str(tier2_agent_round1)
                    prompt = prompt + '\n\nYour previous evaluation:' + '\nSource: ' + src + 'Translation: ' + hyp
                    prompt += 'Error Type: ' + self.tier2_agent_type + '\nError Exist: ' + process[2]["Error Exist"] + '\nSeverity: ' + process[2]["Severity"]
                    prompt += f'\n\nChat History: \nSuperior evaluator: \n{process[3]["round0"]}'
                    message_lst = [
                        {"role": "system", "content": tier2_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    prompt = 'None'
                    message_lst = [
                        {"role": "system", "content": tier2_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                count += 1
                messages_lst.append(message_lst)
            save_json(messages_lst, output_file_path)

    def generate_prompt_tier1_agent_round2(self):
        prompt_description = read_json(self.agent_type_file_path)
        for line in prompt_description:
            if line['description'] == 'tier1_agent_system_prompt':
                tier1_agent_system_prompt = line['content']
            if line['description'] == 'error_severity':
                error_severity = line['content']
            if line['description'] == f'{self.tier1_agent_type}':
                tier1_agent_type_description = line['content']
            if line['description'] == f'{self.tier2_agent_type}':
                tier2_agent_type_description = line['content']
            if line['description'] == f'tier1_agent_round2':
                tier1_agent_round2 = line['content']
        with open(os.path.join(self.dataset_file_path, 'source.txt'), 'r', encoding='utf-8') as srcs, \
            open(os.path.join(self.dataset_file_path, 'hyp.txt'), 'r', encoding='utf-8') as hyps:
            output_file_path = self.messages_path
            process_log_lines = read_json(self.process_log_path)
            messages_lst = []
            count = 0
            for (src, hyp, process) in zip(srcs, hyps, process_log_lines):
                if self.probs_lst[count] == True:
                    prompt = str(tier1_agent_type_description) + '\n' + str(error_severity) + str(tier1_agent_round2)
                    prompt = prompt + '\nThe subcategory evaluator\'s evaluation: ' + '\nSource: ' + src + 'Translation: ' + hyp
                    prompt += 'Error Type: ' + self.tier2_agent_type + '\nError Exist: ' + process[2]["Error Exist"] + '\nSeverity: ' + process[2]["Severity"]
                    prompt += f'\n\nChat History: \nYou: \n{process[3]["round0"]}\nSub-category: \n{process[3]["round1"]}'
                    message_lst = [
                        {"role": "system", "content": tier1_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    prompt = 'None'
                    message_lst = [
                        {"role": "system", "content": tier1_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                count += 1
                messages_lst.append(message_lst)
            save_json(messages_lst, output_file_path)

    def generate_prompt_tier2_agent_round2(self):
        prompt_description = read_json(self.agent_type_file_path)
        for line in prompt_description:
            if line['description'] == 'tier2_agent_system_prompt':
                tier2_agent_system_prompt = line['content']
            if line['description'] == 'error_severity':
                error_severity = line['content']
            if line['description'] == f'{self.tier1_agent_type}':
                tier1_agent_type_description = line['content']
            if line['description'] == f'{self.tier2_agent_type}':
                tier2_agent_type_description = line['content']
            if line['description'] == f'tier2_agent_round2':
                tier2_agent_round2 = line['content']
        with open(os.path.join(self.dataset_file_path, 'source.txt'), 'r', encoding='utf-8') as srcs, \
            open(os.path.join(self.dataset_file_path, 'hyp.txt'), 'r', encoding='utf-8') as hyps:
            output_file_path = self.messages_path
            process_log_lines = read_json(self.process_log_path)
            messages_lst = []
            count = 0
            for (src, hyp, process) in zip(srcs, hyps, process_log_lines):
                if self.probs_lst[count] == True:
                    prompt = f'You are responsible for {self.tier2_agent_type}. The definition of {self.tier2_agent_type} is: ' + str(tier2_agent_type_description) + str(error_severity) + str(tier2_agent_round2)
                    prompt = prompt + '\n\nYour previous evaluation:' + '\nSource: ' + src + 'Translation: ' + hyp
                    prompt += 'Error Type: ' + self.tier2_agent_type + '\nError Exist: ' + process[2]["Error Exist"] + '\nSeverity: ' + process[2]["Severity"]
                    prompt += f'\n\nChat History: \nSuperior evaluator: \n{process[3]["round0"]}\nYou: \n{process[3]["round1"]}\nSuperior evaluator: \n{process[3]["round2"]}'
                    message_lst = [
                        {"role": "system", "content": tier2_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    prompt = 'None'
                    message_lst = [
                        {"role": "system", "content": tier2_agent_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                count += 1
                messages_lst.append(message_lst)
            save_json(messages_lst, output_file_path)