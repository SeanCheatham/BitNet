import argparse
import logging

from common import run_command, find_binary


def run_benchmark():
    bench_path = find_binary("llama-bench")
    command = [
        bench_path,
        '-m', args.model,
        '-n', str(args.n_token),
        '-ngl', '0',
        '-b', '1',
        '-t', str(args.threads),
        '-p', str(args.n_prompt),
        '-r', '5',
    ]
    run_command(command)


def parse_args():
    parser = argparse.ArgumentParser(description='Run end-to-end benchmark')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=True)
    parser.add_argument("-n", "--n-token", type=int, help="Number of generated tokens", required=False, default=128)
    parser.add_argument("-p", "--n-prompt", type=int, help="Prompt to generate text from", required=False, default=512)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_benchmark()
