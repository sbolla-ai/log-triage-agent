"""
Phase 1: A minimal API call to Claude.
Goal: see the request/response shape with our own eyes.
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads .env into os.environ

client = Anthropic()  # auto-reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Say hello and tell me what model you are in one sentence."}
    ],
)

# Inspect the raw response shape — this is the important part
print("=" * 60)
print(f"stop_reason: {response.stop_reason}")
print(f"usage: input={response.usage.input_tokens} output={response.usage.output_tokens}")
print(f"content blocks: {len(response.content)}")
print("-" * 60)
for i, block in enumerate(response.content):
    print(f"Block {i}: type={block.type}")
    if block.type == "text":
        print(f"  text: {block.text}")
print("=" * 60)