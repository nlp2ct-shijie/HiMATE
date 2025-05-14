import json
import os

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

class MultiAgentPrompting_SRc:
    def __init__(self, dataset, lang_pair, translation_system, agent_type, dataset_prefix, agent_type_prefix, error_exist_lst, messages_path, process_logpath):
        '''
        dataset: the different dataset of WMT General MT shared task, such as wmt20, wmt22
        lang_pair: the language pari to be evaluated, such as zhen, deen
        translation_system: the system that provide the translation which will be evaluated by ours framework, such as HuaweiTSC
        agnet_type: the agents responsible for different types of MQM errors, such as accuracy, fluency
        dataset_prefix: the path prefix of the used dataset
        agent_type_prefix: the path prefix of the used prompt which includes the definition of agent types
        error_exist_lst: the list about whether the error exists
        messages_path: the path of the messages to be saved
        process_log_path: The processed log path
        '''
        self.lang_pair = lang_pair
        self.dataset = dataset
        self.translation_system = translation_system
        self.agent_type = agent_type
        self.error_exist_lst = error_exist_lst
        self.messages_path = messages_path
        self.process_logpath = process_logpath

        self.dataset_file_path = os.path.join(dataset_prefix, dataset, lang_pair, translation_system)
        self.agent_type_file_path = os.path.join(agent_type_prefix, lang_pair, f'description_SRc.json')

    def generate_prompt(self):
        prompt_description = read_json(self.agent_type_file_path)
        for line in prompt_description:
            if line['description'] == 'system_prompt':
                system_prompt = line['content']
            if line['description'] == 'error_severity':
                error_severity = line['content']
            if line['description'] == 'reply_format':
                reply_format = line['content']
            if line['description'] == f'{self.agent_type}':
                agent_type_description = line['content']
        with open(os.path.join(self.dataset_file_path, 'source.txt'), 'r', encoding='utf-8') as srcs, \
            open(os.path.join(self.dataset_file_path, 'hyp.txt'), 'r', encoding='utf-8') as hyps:
            output_file_path = self.messages_path
            messages_lst = []
            process_log_lines = read_json(self.process_logpath)
            for (src, hyp, error_exist, process) in zip(srcs, hyps, self.error_exist_lst, process_log_lines):
                if error_exist[0] == 'Yes':
                    prompt = 'Source: ' + src
                    prompt += 'Translation: ' + hyp
                    prompt += 'Existing error type and definition: ' + str(agent_type_description)
                    prompt += 'Severity: ' + error_exist[1] + '(' + str(error_severity) + ')' + '\n'
                    prompt += 'Explanation: ' + process[0]["Explanation"] + '\n'
                    prompt += reply_format
                    message_lst = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    prompt = 'None'
                    message_lst = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                messages_lst.append(message_lst)
            save_json(messages_lst, output_file_path)
