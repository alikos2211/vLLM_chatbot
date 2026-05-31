FROM vllm/vllm-openai:latest

COPY serve.py /app/serve.py

EXPOSE 8000

ENV MODEL_NAME=gpt2
ENV HOST=0.0.0.0
ENV PORT=8000

ENTRYPOINT ["python3", "/app/serve.py"]