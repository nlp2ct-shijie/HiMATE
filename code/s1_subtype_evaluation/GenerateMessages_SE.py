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

class MultiAgentPrompting_SE:
    def __init__(self, dataset, lang_pair, translation_system, agent_type, dataset_prefix, agent_type_prefix, messages_path):
        '''
        dataset: the different dataset of WMT General MT shared task, such as wmt20, wmt22
        lang_pair: the language pari to be evaluated, such as zhen, deen
        translation_system: the system that provide the translation which will be evaluated by ours framework, such as HuaweiTSC
        agnet_type: the agents responsible for different types of MQM errors, such as accuracy, fluency
        dataset_prefix: the path prefix of the used dataset
        agent_type_prefix: the path prefix of the used prompt which includes the definition of agent types
        messages_path: the path of the messages to be saved
        '''
        self.lang_pair = lang_pair
        self.dataset = dataset
        self.translation_system = translation_system
        self.agent_type = agent_type
        self.messages_path = messages_path

        self.dataset_file_path = os.path.join(dataset_prefix, dataset, lang_pair, translation_system)
        self.agent_type_file_path = os.path.join(agent_type_prefix, lang_pair, f'description_SE.json')

    def generate_prompt(self):
        prompt_description = read_json(self.agent_type_file_path)
        for line in prompt_description:
            if line['description'] == 'system_prompt':
                system_prompt = line['content']
            if line['description'] == 'answer_requirment_more_error_exist':
                answer_requirment = line['content']
            if line['description'] == 'error_severity':
                error_severity = line['content']
            if line['description'] == 'reply_format':
                reply_format = line['content']
            if line['description'] == f'{self.agent_type}':
                agent_type_description = line['content']
            if line['description'] == f'{self.agent_type}_example_1':
                agent_example_1 = line['content']
            if line['description'] == f'{self.agent_type}_example_2':
                agent_example_2 = line['content']
        with open(os.path.join(self.dataset_file_path, 'source.txt'), 'r', encoding='utf-8') as srcs, \
            open(os.path.join(self.dataset_file_path, 'hyp.txt'), 'r', encoding='utf-8') as hyps:
            output_file_path = self.messages_path
            messages_lst = []
            for (src, hyp) in zip(srcs, hyps):
                prompt = str(agent_type_description) + str(answer_requirment) + str(error_severity)
                prompt = prompt + '\nSource: ' + src + 'Translation: ' + hyp
                prompt += '\n' + str(reply_format) + '\n' + str(agent_example_1) + str(agent_example_2)
                message_lst = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                messages_lst.append(message_lst)
            save_json(messages_lst, output_file_path)
