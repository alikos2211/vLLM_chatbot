import os
import logging
import argparse
import asyncio
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    model = os.environ.get("MODEL_NAME", "gpt2")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    dtype = os.environ.get("DTYPE", "float16")

    log.info(f"Starting vLLM with model: {model}")
    
    
    parser = make_arg_parser(argparse.ArgumentParser())
    args = parser.parse_args([
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--dtype", dtype,
        "--max-model-len", "1024",
    ])

    asyncio.run(run_server(args))