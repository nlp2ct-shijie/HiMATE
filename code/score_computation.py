import json
import argparse
import os

# List of error agent types to process
agent_lst = [
    'Addition', 'Omission', 'Mistranslation', 'Untranslated_text',
    'Inappropriate_for_context', 'Inconsistent_use', 'Punctuation',
    'Spelling', 'Grammar', 'Register', 'Inconsistency', 'Character_encoding',
    'Awkward', 'Address_format', 'Currency_format', 'Date_format',
    'Name_format', 'Telephone_format', 'Time_format'
]


def read_json(path):
    """Read and parse JSON file from given path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_sentence_scores(log_dir, output_file, stage):
    """
    Calculate total error scores for each sentence across all agent types.

    Args:
        log_dir: Path to directory containing log files
        output_file: Target output text file
        stage: Processing stage to analyze (e.g., 'CD')
    """
    # Determine total number of sentences from the first available log file
    num_sentences = 0
    for agent in agent_lst:
        log_path = os.path.join(log_dir, f'{agent}.process.json')
        if os.path.exists(log_path):
            num_sentences = len(read_json(log_path))
            break

    # Initialize scores for all sentences
    sentence_totals = [0.0 for _ in range(num_sentences)]

    # Accumulate scores from all agents
    for agent in agent_lst:
        log_path = os.path.join(log_dir, f'{agent}.process.json')
        if not os.path.exists(log_path):
            continue

        log_data = read_json(log_path)
        for sentence_idx, sentence_logs in enumerate(log_data):
            if sentence_idx >= num_sentences:
                break  # Safety check for inconsistent data

            # Calculate penalty for this agent and sentence
            agent_penalty = sum(
                compute_score(
                    item['Error Exist'],
                    item['Severity'],
                    agent
                )
                for item in sentence_logs
                if str(item['stage']) == stage
            )

            # Update sentence total (negative cumulative penalty)
            sentence_totals[sentence_idx] -= agent_penalty

    # Write results to text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for total in sentence_totals:
            f.write(f"{total}\n")


def compute_score(error_exist, error_severity, agent_type):
    """
    Calculate penalty score for a single error entry.

    Returns:
        float: Positive penalty value for errors
    """
    if 'Yes' in error_exist:
        if 'Major' in error_severity:
            return 5
        elif 'Minor' in error_severity:
            return 0.1 if 'Punctuation' in agent_type else 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Calculate scores from log files'
    )
    parser.add_argument(
        '-l', '--log-path',
        help='Directory containing agent log files'
    )
    parser.add_argument(
        '-s', '--stage',
        help='Target processing stage (e.g., CD)'
    )
    parser.add_argument(
        '-o', '--output-file',
        help='Output text file for sentence scores'
    )

    args = parser.parse_args()
    compute_sentence_scores(args.log_path, args.output_file, args.stage)


if __name__ == '__main__':
    main()