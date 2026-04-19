"""
Phase 1b: Proof of statelessness.
Run two separate API calls and show the LLM has no memory between them.
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# Call 1
print("--- CALL 1 ---")
r1 = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "My favorite color is teal. Remember this."}],
)
print(f"Claude says: {r1.content[0].text}")

# Call 2 — completely fresh, no history passed
print("\n--- CALL 2 (no history) ---")
r2 = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is my favorite color?"}],
)
print(f"Claude says: {r2.content[0].text}")

# Call 3 — same question, but we pass the full history
print("\n--- CALL 3 (with history) ---")
r3 = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "My favorite color is teal. Remember this."},
        {"role": "assistant", "content": r1.content[0].text},
        {"role": "user", "content": "What is my favorite color?"},
    ],
)
print(f"Claude says: {r3.content[0].text}")