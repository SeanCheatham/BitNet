import subprocess
import sys
import os
import platform
import argparse
import logging
import shutil
from pathlib import Path

from utils.common import (
    SUPPORTED_HF_MODELS,
    SUPPORTED_QUANT_TYPES,
    COMPILER_EXTRA_ARGS,
    OS_EXTRA_ARGS,
    system_info,
    run_command,
    find_binary,
    setup_signal_handler,
)

logger = logging.getLogger("setup_env")


def get_model_name():
    if args.hf_repo:
        return SUPPORTED_HF_MODELS[args.hf_repo]
    return os.path.basename(os.path.normpath(args.model_dir))


def prepare_model():
    hf_url = args.hf_repo
    model_dir = args.model_dir
    quant_type = args.quant_type
    quant_embd = args.quant_embd
    if hf_url is not None:
        model_dir = os.path.join(model_dir, SUPPORTED_HF_MODELS[hf_url])
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model {hf_url} from HuggingFace to {model_dir}...")
        run_command(["huggingface-cli", "download", hf_url, "--local-dir", model_dir], log_step="download_model", log_dir=args.log_dir)
    elif not os.path.exists(model_dir):
        logging.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)
    else:
        logging.info(f"Loading model from directory {model_dir}.")
    gguf_path = os.path.join(model_dir, "ggml-model-" + quant_type + ".gguf")
    if not os.path.exists(gguf_path) or os.path.getsize(gguf_path) == 0:
        logging.info(f"Converting HF model to GGUF format...")
        if quant_type.startswith("tl"):
            run_command([sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", quant_type, "--quant-embd"], log_step="convert_to_tl", log_dir=args.log_dir)
        else: # i2s
            run_command([sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", "f32"], log_step="convert_to_f32_gguf", log_dir=args.log_dir)
            f32_model = os.path.join(model_dir, "ggml-model-f32.gguf")
            i2s_model = os.path.join(model_dir, "ggml-model-i2_s.gguf")
            quantize = find_binary("llama-quantize")
            if quant_embd:
                run_command([quantize, "--token-embedding-type", "f16", f32_model, i2s_model, "I2_S", "1", "1"], log_step="quantize_to_i2s", log_dir=args.log_dir)
            else:
                run_command([quantize, f32_model, i2s_model, "I2_S", "1"], log_step="quantize_to_i2s", log_dir=args.log_dir)

        logging.info(f"GGUF model saved at {gguf_path}")
    else:
        logging.info(f"GGUF model already exists at {gguf_path}")


def setup_gguf():
    run_command([sys.executable, "-m", "pip", "install", "3rdparty/llama.cpp/gguf-py"], log_step="install_gguf", log_dir=args.log_dir)


def gen_code():
    _, arch = system_info()

    llama3_f3_models = set([name for name in SUPPORTED_HF_MODELS.values() if name.startswith("Falcon") or name.startswith("Llama")])

    if arch == "arm64":
        if args.use_pretuned:
            pretuned_kernels = os.path.join("preset_kernels", get_model_name())
            if not os.path.exists(pretuned_kernels):
                logging.error(f"Pretuned kernels not found for model {args.hf_repo}")
                sys.exit(1)
            if args.quant_type == "tl1":
                shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl1.h"), "include/bitnet-lut-kernels.h")
                shutil.copyfile(os.path.join(pretuned_kernels, "kernel_config_tl1.ini"), "include/kernel_config.ini")
            elif args.quant_type == "tl2":
                shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl2.h"), "include/bitnet-lut-kernels.h")
                shutil.copyfile(os.path.join(pretuned_kernels, "kernel_config_tl2.ini"), "include/kernel_config.ini")
        if get_model_name() == "bitnet_b1_58-large":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "128,64,128", "--bm", "32,64,32"], log_step="codegen", log_dir=args.log_dir)
        elif get_model_name() in llama3_f3_models:
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "Llama3-8B-1.58-100B-tokens", "--BM", "256,128,256,128", "--BK", "128,64,128,64", "--bm", "32,64,32,64"], log_step="codegen", log_dir=args.log_dir)
        elif get_model_name() == "bitnet_b1_58-3B":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "64,128,64", "--bm", "32,64,32"], log_step="codegen", log_dir=args.log_dir)
        elif get_model_name() == "BitNet-b1.58-2B-4T":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "BitNet-2B-4T", "--BM", "128,128,256,128", "--BK", "64,64,128,64", "--bm", "32,32,64,32"], log_step="codegen", log_dir=args.log_dir)
        else:
            raise NotImplementedError(f"Model '{get_model_name()}' is not supported for codegen on {arch}")
    else:
        if args.use_pretuned:
            pretuned_kernels = os.path.join("preset_kernels", get_model_name())
            if not os.path.exists(pretuned_kernels):
                logging.error(f"Pretuned kernels not found for model {args.hf_repo}")
                sys.exit(1)
            shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl2.h"), "include/bitnet-lut-kernels.h")
        if get_model_name() == "bitnet_b1_58-large":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "96,192,96", "--bm", "32,32,32"], log_step="codegen", log_dir=args.log_dir)
        elif get_model_name() in llama3_f3_models:
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "Llama3-8B-1.58-100B-tokens", "--BM", "256,128,256,128", "--BK", "96,96,96,96", "--bm", "32,32,32,32"], log_step="codegen", log_dir=args.log_dir)
        elif get_model_name() == "bitnet_b1_58-3B":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32"], log_step="codegen", log_dir=args.log_dir)
        elif get_model_name() == "BitNet-b1.58-2B-4T":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "BitNet-2B-4T", "--BM", "128,128,128,128", "--BK", "96,96,96,96", "--bm", "32,32,32,32"], log_step="codegen", log_dir=args.log_dir)
        else:
            raise NotImplementedError(f"Model '{get_model_name()}' is not supported for codegen on {arch}")


def compile():
    cmake_exists = subprocess.run(["cmake", "--version"], capture_output=True)
    if cmake_exists.returncode != 0:
        logging.error("Cmake is not available. Please install CMake and try again.")
        sys.exit(1)
    _, arch = system_info()
    if arch not in COMPILER_EXTRA_ARGS:
        logging.error(f"Arch {arch} is not supported yet")
        sys.exit(1)
    logging.info("Compiling the code using CMake.")
    run_command(["cmake", "-B", "build", *COMPILER_EXTRA_ARGS[arch], *OS_EXTRA_ARGS.get(platform.system(), []), "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"], log_step="generate_build_files", log_dir=args.log_dir)
    run_command(["cmake", "--build", "build", "--config", "Release"], log_step="compile", log_dir=args.log_dir)


def main():
    setup_gguf()
    gen_code()
    compile()
    prepare_model()


def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(description='Setup the environment for running the inference')
    parser.add_argument("--hf-repo", "-hr", type=str, help="Model used for inference", choices=SUPPORTED_HF_MODELS.keys())
    parser.add_argument("--model-dir", "-md", type=str, help="Directory to save/load the model", default="models")
    parser.add_argument("--log-dir", "-ld", type=str, help="Directory to save the logging info", default="logs")
    parser.add_argument("--quant-type", "-q", type=str, help="Quantization type", choices=SUPPORTED_QUANT_TYPES[arch], default="i2_s")
    parser.add_argument("--quant-embd", action="store_true", help="Quantize the embeddings to f16")
    parser.add_argument("--use-pretuned", "-p", action="store_true", help="Use the pretuned kernel parameters")
    return parser.parse_args()


if __name__ == "__main__":
    setup_signal_handler()
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    main()
