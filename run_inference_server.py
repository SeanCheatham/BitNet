import argparse
import logging

from utils.common import run_command, find_binary, setup_signal_handler


def run_server():
    server_path = find_binary("llama-server")
    command = [
        server_path,
        '-m', args.model,
        '-c', str(args.ctx_size),
        '-t', str(args.threads),
        '-n', str(args.n_predict),
        '-ngl', '0',
        '--temp', str(args.temperature),
        '--host', args.host,
        '--port', str(args.port),
        '-cb',
    ]
    if args.prompt:
        command.extend(['-p', args.prompt])

    logging.info(f"Starting server on {args.host}:{args.port}")
    run_command(command)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_signal_handler()
    parser = argparse.ArgumentParser(description='Run llama.cpp server')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str, help="System prompt for the model", required=False)
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", required=False, default=4096)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the context window", required=False, default=2048)
    parser.add_argument("--temperature", type=float, help="Temperature for sampling", required=False, default=0.8)
    parser.add_argument("--host", type=str, help="IP address to listen on", required=False, default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Port to listen on", required=False, default=8080)

    args = parser.parse_args()
    run_server()
