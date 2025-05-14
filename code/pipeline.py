from openai import OpenAI
import json
import argparse
from tqdm import tqdm
import os
import re
import numpy as np

from s1_subtype_evaluation.CallModel_SE import CallModel_SE
from s1_subtype_evaluation.GenerateMessages_SE import MultiAgentPrompting_SE
from s2_self_reflection_correction.CallModel_SRc import CallModel_SRc
from s2_self_reflection_correction.GenerateMessages_SRc import MultiAgentPrompting_SRc
from s3_self_reflection_verification.CallModel_SRv import CallModel_SRv
from s3_self_reflection_verification.GenerateMessages_SRv import MultiAgentPrompting_SRv
from s4_collaborative_discussion.CallModel_CD import CallModel_CD
from s4_collaborative_discussion.GenerateMessages_CD import MultiAgentPrompting_CD
from get_lst import update_probs_lst, get_error_exist_lst
from record import record_SE, record_SRc, record_SRv, record_CD, record_final


def main():
    parser = argparse.ArgumentParser('Command-line script')
    parser.add_argument('-l', '--lang', type=str, default='zhen',
                        help='the language pari to be evaluated, such as zhen, deen')
    parser.add_argument('-d', '--dataset', type=str, default='wmt22',
                        help='the different dataset of WMT General MT shared task, such as wmt21, wmt22')
    parser.add_argument('-m', '--model', type=str, default='gpt-4',
                        help='the model endpoint used for evaluation')
    parser.add_argument('--system', type=str, default='',
                        help='the system that provide the translation which will be evaluated by ours framework, such as HuaweiTSC')
    parser.add_argument('--agent-type', type=str, default='',
                        help='the agents responsible for different types of MQM errors, such as accuracy, fluency')
    parser.add_argument('--dataset-prefix', type=str, default='',
                        help='the file path prefix of the dataset')
    parser.add_argument('--agent-type-prefix', type=str, default='',
                        help='the file path prefix of the agent type')
    parser.add_argument('-k', '--key', type=str, required=True,
                        help='api key of model')
    parser.add_argument('--url', type=str, required=True,
                        help='base url of model')
    parser.add_argument('-t', '--temperature', type=float, default=0,
                        help='temperature')
    parser.add_argument('--max-tokens', type=int, default=500,
                        help='max tokens')
    parser.add_argument('--next-id', type=int, default=0,
                        help='the next id of sentence to train')
    args = parser.parse_args()

    logpath = os.path.join('../logs', args.dataset, args.lang, args.system)
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    process_logpath = os.path.join(logpath, f'{args.agent_type}.process.json')

    # Subtype Evaluation
    messages_path_SE = os.path.join('../messages', args.dataset, args.lang, args.system,
                                        f's1_SubtyepEvaluation/{args.agent_type}.json')

    responses_path_prefix_SE = os.path.join('../responses', args.dataset, args.lang, args.system, 's1_SubtyepEvaluation')
    if not os.path.exists(responses_path_prefix_SE):
        os.makedirs(responses_path_prefix_SE)

    responses_path_SE = os.path.join(responses_path_prefix_SE, f'{args.agent_type}.json')
    responses_log_path_SE = os.path.join(responses_path_prefix_SE, f'log.{args.agent_type}.json')

    # Generate messages
    MAP_SE = MultiAgentPrompting_SE(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix, args.agent_type_prefix, messages_path_SE)
    MAP_SE.generate_prompt()
    
    # Responses
    CCG_SE = CallModel_SE(args.model, messages_path_SE, responses_path_SE, responses_log_path_SE, args.url, args.key, args.next_id, args.max_tokens, args.temperature)
    CCG_SE.call_model()

    record_SE(process_logpath, responses_path_SE)



    # Self-Reflection correction
    error_exist_lst = get_error_exist_lst(process_logpath)

    messages_path_SRc = os.path.join('../messages', args.dataset, args.lang, args.system,
                                        f's2_Self_Reflection_correction/{args.agent_type}.json')

    responses_path_prefix_SRc = os.path.join('../responses', args.dataset, args.lang, args.system, 's2_Self_Reflection_correction')
    if not os.path.exists(responses_path_prefix_SRc):
        os.makedirs(responses_path_prefix_SRc)

    responses_path_SRc = os.path.join(responses_path_prefix_SRc, f'{args.agent_type}.json')
    responses_log_path_SRc = os.path.join(responses_path_prefix_SRc, f'log.{args.agent_type}.json')

    MAP_SRc = MultiAgentPrompting_SRc(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix,
                                      args.agent_type_prefix, error_exist_lst, messages_path_SRc, process_logpath)
    MAP_SRc.generate_prompt()
    
    CCG_SRc = CallModel_SRc(args.model, messages_path_SRc, responses_path_SRc, responses_log_path_SRc, args.url, args.key,
                                   error_exist_lst, args.next_id, args.max_tokens, args.temperature, logprobs=False)
    CCG_SRc.call_model()

    record_SRc(process_logpath, responses_path_SRc, error_exist_lst)


    # Self-Reflection verification
    error_exist_lst = get_error_exist_lst(process_logpath)

    messages_path_SRv = os.path.join('../messages', args.dataset, args.lang, args.system,
                                        f's3_Self_Reflection_verification/{args.agent_type}.json')

    responses_path_prefix_SRv = os.path.join('../responses', args.dataset, args.lang, args.system, 's3_Self_Reflection_verification')
    if not os.path.exists(responses_path_prefix_SRv):
        os.makedirs(responses_path_prefix_SRv)

    responses_path_SRv = os.path.join(responses_path_prefix_SRv, f'{args.agent_type}.json')
    responses_log_path_SRv = os.path.join(responses_path_prefix_SRv, f'log.{args.agent_type}.json')

    MAP_SRv = MultiAgentPrompting_SRv(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix,
                                           args.agent_type_prefix, error_exist_lst, messages_path_SRv, process_logpath)
    MAP_SRv.generate_prompt()
    
    CCG_SRv = CallModel_SRv(args.model, messages_path_SRv, responses_path_SRv, responses_log_path_SRv, args.url, args.key,
                                   error_exist_lst, args.next_id, args.max_tokens, args.temperature, logprobs=True)
    CCG_SRv.call_model()

    record_SRv(process_logpath, responses_path_SRv, error_exist_lst)
    
    # Collaborative Discussion
    probs_lst = []
    for round in range(4):
        probs_lst = update_probs_lst(process_logpath, probs_lst, round, args.agent_type, args.model, args.lang)

        if round == 0:
            messages_path_CD = os.path.join('../messages', args.dataset, args.lang, args.system, 'round0',
                                                f's4_Collaborative_Discussion/{args.agent_type}.json')

            responses_path_prefix_CD = os.path.join('../responses', args.dataset, args.lang, args.system, 's4_Collaborative_Discussion/round0')
            if not os.path.exists(responses_path_prefix_CD):
                os.makedirs(responses_path_prefix_CD)

            responses_path_CD = os.path.join(responses_path_prefix_CD, f'{args.agent_type}.json')
            responses_log_path_CD = os.path.join(responses_path_prefix_CD, f'log.{args.agent_type}.json')

            MAP_CD = MultiAgentPrompting_CD(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix,
                                                   args.agent_type_prefix, probs_lst, messages_path_CD, process_logpath)
            MAP_CD.generate_prompt_tier1_agent_round1()


            CCG_CD = CallModel_CD(args.model, messages_path_CD, responses_path_CD, responses_log_path_CD, args.url, args.key,
                                    probs_lst, args.next_id, args.max_tokens, args.temperature)
            CCG_CD.call_model()

            record_CD(process_logpath, responses_path_CD, round)

        elif round == 1:
            messages_path_CD = os.path.join('../messages', args.dataset, args.lang, args.system, 'round1',
                                                f's4_Collaborative_Discussion/{args.agent_type}.json')

            responses_path_prefix_CD = os.path.join('../responses', args.dataset, args.lang, args.system, 's4_Collaborative_Discussion/round1')
            if not os.path.exists(responses_path_prefix_CD):
                os.makedirs(responses_path_prefix_CD)

            responses_path_CD = os.path.join(responses_path_prefix_CD, f'{args.agent_type}.json')
            responses_log_path_CD = os.path.join(responses_path_prefix_CD, f'log.{args.agent_type}.json')

            MAP_CD = MultiAgentPrompting_CD(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix,
                                            args.agent_type_prefix, probs_lst, messages_path_CD, process_logpath)
            MAP_CD.generate_prompt_tier2_agent_round1()

            CCG_CD = CallModel_CD(args.model, messages_path_CD, responses_path_CD, responses_log_path_CD, args.url,
                                    args.key, probs_lst, args.next_id, args.max_tokens, args.temperature)
            CCG_CD.call_model()

            record_CD(process_logpath, responses_path_CD, round)

        elif round == 2:
            messages_path_CD = os.path.join('../messages', args.dataset, args.lang, args.system, 'round2',
                                                f's4_Collaborative_Discussion/{args.agent_type}.json')

            responses_path_prefix_CD = os.path.join('../responses', args.dataset, args.lang, args.system, 's4_Collaborative_Discussion/round2')
            if not os.path.exists(responses_path_prefix_CD):
                os.makedirs(responses_path_prefix_CD)

            responses_path_CD = os.path.join(responses_path_prefix_CD, f'{args.agent_type}.json')
            responses_log_path_CD = os.path.join(responses_path_prefix_CD, f'log.{args.agent_type}.json')

            MAP_CD = MultiAgentPrompting_CD(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix,
                                            args.agent_type_prefix, probs_lst, messages_path_CD, process_logpath)
            MAP_CD.generate_prompt_tier1_agent_round2()

            CCG_CD = CallModel_CD(args.model, messages_path_CD, responses_path_CD, responses_log_path_CD, args.url,
                                    args.key, probs_lst, args.next_id, args.max_tokens, args.temperature)
            CCG_CD.call_model()

            record_CD(process_logpath, responses_path_CD, round)

        elif round == 3:
            messages_path_CD = os.path.join('../messages', args.dataset, args.lang, args.system, 'round3',
                                                f's4_Collaborative_Discussion/{args.agent_type}.json')

            responses_path_prefix_CD = os.path.join('../responses', args.dataset, args.lang, args.system, 's4_Collaborative_Discussion/round3')
            if not os.path.exists(responses_path_prefix_CD):
                os.makedirs(responses_path_prefix_CD)

            responses_path_CD = os.path.join(responses_path_prefix_CD, f'{args.agent_type}.json')
            responses_log_path_CD = os.path.join(responses_path_prefix_CD, f'log.{args.agent_type}.json')

            MAP_CD = MultiAgentPrompting_CD(args.dataset, args.lang, args.system, args.agent_type, args.dataset_prefix,
                                            args.agent_type_prefix, probs_lst, messages_path_CD, process_logpath)
            MAP_CD.generate_prompt_tier2_agent_round2()

            CCG_CD = CallModel_CD(args.model, messages_path_CD, responses_path_CD, responses_log_path_CD, args.url,
                                    args.key, probs_lst, args.next_id, args.max_tokens, args.temperature)
            CCG_CD.call_model()

            record_CD(process_logpath, responses_path_CD, round)

    record_final(process_logpath)

    
if __name__ == '__main__':
    main()
