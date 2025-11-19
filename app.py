import dotenv
dotenv.load_dotenv()

import gradio as gr

from google import genai
from google.genai import types
client = genai.Client()

import time
while True:
    time.sleep(1)
    file_search_stores = client.file_search_stores.list()
    if file_search_stores:
        file_search_store = file_search_stores[0]
        break

# Patch Starlette middleware to handle the ASGI message issue
# This fixes the "Unexpected message" AssertionError in MCP server
import starlette.middleware.base

_original_call = starlette.middleware.base.BaseHTTPMiddleware.__call__

async def _patched_call(self, scope, receive, send):
    """Patched version that catches and handles ASGI message errors."""
    try:
        await _original_call(self, scope, receive, send)
    except (AssertionError, Exception) as e:
        error_msg = str(e)
        if "Unexpected message" in error_msg and "http.response.start" in error_msg:
            # Known issue with Gradio MCP server - log and continue
            print(f"[MCP Server] Caught and handled ASGI middleware issue: {error_msg[:100]}")
            return
        raise

starlette.middleware.base.BaseHTTPMiddleware.__call__ = _patched_call


def answer(message: str, history: list = None):
    """Answer questions about MacroMicro internal Standard Operating Procedures (SOP).

    Uses FileSearch to retrieve relevant information from the SOP documentation
    and provides detailed answers to help team members understand workflows and procedures.

    Args:
        message: The current input message from the user (string when type="messages")
        history: Chat history in Gradio format (list of dicts with "role" and "content" keys). Optional, defaults to None.

    Yields:
        Detailed answer based on retrieved SOP documentation (dict with "role" and "content" when type="messages")
    """
    # Convert Gradio messages format to Gemini API format
    # Gradio format: [{"role": "user" | "assistant", "content": str}, ...]
    # Gemini format: [{"role": "user" | "model", "parts": [{"text": str}]}, ...]

    gemini_contents = []

    # Handle history parameter - ensure it's a list
    if history and isinstance(history, list):
        for msg in history:
            # Skip invalid messages
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Convert Gradio "assistant" role to Gemini "model" role
            gemini_role = "model" if role == "assistant" else "user"

            gemini_contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

    # Add the current user message
    if message:
        gemini_contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })

    if not gemini_contents:
        yield {"role": "assistant", "content": "Please provide a question."}
        return

    # Stream the response for better UX
    try:
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction="你的任務：依據FileSearch工具檢索到的資料，詳細回答MacroMicro團隊內部標準作業流程（SOP）相關問題",
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[file_search_store.name]
                        )
                    )
                ]
            )
        )
    except Exception as e:
        print(f"Error creating response stream: {e}")
        yield {"role": "assistant", "content": f"Error connecting to AI service: {str(e)}"}
        return

    # Stream response chunks as they arrive
    accumulated_text = ""
    has_yielded = False

    try:
        for chunk in response_stream:
            # Handle chunks that may contain non-text parts (e.g., executable_code)
            chunk_text = ""
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                chunk_text += part.text

            if chunk_text:
                print(f"Streamed chunk: {chunk_text}")
                accumulated_text += chunk_text
                has_yielded = True
                # Yield in Gradio messages format (required for type="messages")
                yield {"role": "assistant", "content": accumulated_text}
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        if not has_yielded:
            yield {"role": "assistant", "content": f"Error generating response: {str(e)}"}
        else:
            yield {"role": "assistant", "content": accumulated_text + f"\n\n[Error occurred during generation: {str(e)}]"}
        return

    # Ensure we always yield at least once
    if not has_yielded:
        yield {"role": "assistant", "content": "No response generated. Please try rephrasing your question."}


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    answer,
    type="messages",
    title="Hi, MMer 👋",
)

with gr.Blocks(fill_height=True, title="MM SOP") as demo:
    chatbot.render()

if __name__ == "__main__":
    demo.launch(mcp_server=True)