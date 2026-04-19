"""
Phase 2: Give Claude a single tool. See what a tool_use response looks like.
No loop yet — we just capture the response and inspect it.
"""
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# ---- Tool definition: the CONTRACT we show to Claude ----
tools = [
    {
        "name": "get_log_stats",
        "description": (
            "Returns high-level statistics about a log file: total line count, "
            "counts per severity level (ERROR, WARN, INFO, DEBUG), and the "
            "timestamp range covered. Use this when the user asks for an "
            "overview or summary of a log file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_file": {
                    "type": "string",
                    "description": "Path to the log file to analyze",
                }
            },
            "required": ["log_file"],
        },
    }
]

# ---- The API call — note the new `tools` parameter ----
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[
        {
            "role": "user",
            "content": "Give me an overview of /var/log/app.log",
        }
    ],
)

# ---- Inspect the response — THIS is what changes ----
print("=" * 60)
print(f"stop_reason: {response.stop_reason}")
print(f"content blocks: {len(response.content)}")
print("-" * 60)

for i, block in enumerate(response.content):
    print(f"Block {i}: type={block.type}")
    if block.type == "text":
        print(f"  text: {block.text}")
    elif block.type == "tool_use":
        print(f"  tool name:  {block.name}")
        print(f"  tool id:    {block.id}")
        print(f"  tool input: {block.input}")

print("=" * 60)