from openai import OpenAI
import json
import argparse
from tqdm import tqdm
import os
import time

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {path}.')
    return

def read_next_id(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)["next_id"]

class CallModel_CD:
    def __init__(self, model, messages_path, responses_path, responses_log_path, base_url, api_key, probs_lst, next_id=0, max_tokens=500,
                 temperature=0, stop=None, seed=42, tools=None, logprobs=None, top_logprobs=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.next_id = next_id
        self.stop = stop
        self.seed = seed
        self.tools = tools
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        self.messages_path = messages_path
        self.responses_path = responses_path
        self.responses_log_path = responses_log_path
        self.probs_lst = probs_lst

        self.base_url = base_url
        self.api_key = api_key

        self.messages = self.read_message()

    def read_message(self):
        messages = read_json(self.messages_path)
        return messages

    def get_completion(self, message):
        params = {
            "messages": message,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": self.stop,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }
        if self.tools:
            params["tools"] = self.tools

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        completion = client.chat.completions.create(**params)
        return completion

    def call_model(self):
        messages_count = len(self.messages)
        state_dict = {
            "next_id": self.next_id,
            "error_message": []
        }
        save_json(state_dict, self.responses_log_path)

        responses = []
        flag = True

        while flag:
            next_id = read_next_id(self.responses_log_path)
            range_max = min(next_id + 5, messages_count)  # Dynamic batch size

            # Manually control the progress bar
            with tqdm(total=range_max - next_id, desc=f"Call {self.model}") as pbar:
                current_i = next_id

                while current_i < range_max:
                    retries = 3  # Independent retry counter for each index
                    last_error = None  # Save the last error message

                    # Only enter retry logic when processing is required
                    if self.probs_lst[current_i] == True:
                        while retries > 0:
                            try:
                                response = self.get_completion(self.messages[current_i])
                                # Exit retry loop immediately upon success
                                response_dict = {
                                    "id": current_i,
                                    "content": response.choices[0].message.content
                                }
                                break  # Exit retry loop when successful
                            except Exception as e:
                                last_error = e  # Temporarily store last error
                                retries -= 1
                                if retries == 0:  # Record only when all three retries fail
                                    error_type = type(last_error).__name__
                                    error_dict = {
                                        "error_id": current_i,
                                        "error_info": f"{error_type}: {last_error}"
                                    }
                                    state_dict["error_message"].append(error_dict)
                                    save_json(state_dict, self.responses_log_path)  # Save only on final failure

                                    # Generate error response
                                    response_dict = {
                                        "id": current_i,
                                        "content": f"Error after 3 retries: {error_type}"
                                    }
                    else:
                        # Process that does not require processing
                        response_dict = {
                            "id": current_i,
                            "content": 'None',
                        }

                    # Record result regardless of success/failure
                    responses.append(response_dict)

                    # Ensure the index always advances
                    current_i += 1
                    pbar.update(1)  # Manually update progress bar

            # Save results of this batch
            save_json(responses, self.responses_path)
            state_dict["next_id"] = range_max
            save_json(state_dict, self.responses_log_path)

            # Check termination condition
            if range_max == messages_count:
                flag = False