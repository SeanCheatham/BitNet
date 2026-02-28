"""Shared utilities for BitNet scripts."""

import logging
import os
import platform
import signal
import subprocess
import sys

logger = logging.getLogger(__name__)

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}

# Maps HuggingFace repo ID to the short model name used internally
SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": "bitnet_b1_58-large",
    "1bitLLM/bitnet_b1_58-3B": "bitnet_b1_58-3B",
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": "Llama3-8B-1.58-100B-tokens",
    "tiiuae/Falcon3-7B-Instruct-1.58bit": "Falcon3-7B-Instruct-1.58bit",
    "tiiuae/Falcon3-7B-1.58bit": "Falcon3-7B-1.58bit",
    "tiiuae/Falcon3-10B-Instruct-1.58bit": "Falcon3-10B-Instruct-1.58bit",
    "tiiuae/Falcon3-10B-1.58bit": "Falcon3-10B-1.58bit",
    "tiiuae/Falcon3-3B-Instruct-1.58bit": "Falcon3-3B-Instruct-1.58bit",
    "tiiuae/Falcon3-3B-1.58bit": "Falcon3-3B-1.58bit",
    "tiiuae/Falcon3-1B-Instruct-1.58bit": "Falcon3-1B-Instruct-1.58bit",
    "microsoft/BitNet-b1.58-2B-4T": "BitNet-b1.58-2B-4T",
    "tiiuae/Falcon-E-3B-Instruct": "Falcon-E-3B-Instruct",
    "tiiuae/Falcon-E-1B-Instruct": "Falcon-E-1B-Instruct",
    "tiiuae/Falcon-E-3B-Base": "Falcon-E-3B-Base",
    "tiiuae/Falcon-E-1B-Base": "Falcon-E-1B-Base",
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"],
}

COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=OFF"],
    "x86_64": ["-DBITNET_X86_TL2=OFF"],
}

OS_EXTRA_ARGS = {
    "Windows": ["-T", "ClangCL"],
}


def system_info():
    """Return (os_name, arch) for the current platform."""
    machine = platform.machine()
    arch = ARCH_ALIAS.get(machine)
    if arch is None:
        logger.error(
            f"Unsupported architecture: {machine}. "
            f"Supported: {', '.join(sorted(set(ARCH_ALIAS.values())))}"
        )
        sys.exit(1)
    return platform.system(), arch


def find_binary(name):
    """Resolve platform-specific path to a built binary.

    Checks build/bin/Release/<name>.exe (Windows) then build/bin/<name>.
    Returns the path string if found, or exits with an error.
    """
    build_dir = "build"
    if platform.system() == "Windows":
        path = os.path.join(build_dir, "bin", "Release", f"{name}.exe")
        if os.path.exists(path):
            return path
    path = os.path.join(build_dir, "bin", name)
    if os.path.exists(path):
        return path
    logger.error(f"Binary '{name}' not found in {build_dir}/bin/. Please build first.")
    sys.exit(1)


def run_command(command, shell=False, log_step=None, log_dir=None):
    """Run a system command and exit on failure.

    Args:
        command: Command list or string to execute.
        shell: Whether to use shell execution.
        log_step: If provided, stdout/stderr are written to <log_dir>/<log_step>.log.
        log_dir: Directory for log files. Required if log_step is set.
    """
    if log_step:
        if log_dir is None:
            raise ValueError("log_dir is required when log_step is set")
        log_file = os.path.join(log_dir, log_step + ".log")
        with open(log_file, "w") as f:
            try:
                subprocess.run(command, shell=shell, check=True, stdout=f, stderr=f)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running command: {e}, check details in {log_file}")
                sys.exit(1)
    else:
        try:
            subprocess.run(command, shell=shell, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running command: {e}")
            sys.exit(1)


def setup_signal_handler():
    """Install a SIGINT handler that exits cleanly."""
    def _handler(sig, frame):
        logger.info("Ctrl+C pressed, exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
