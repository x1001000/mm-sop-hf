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


def answer(message, history=None):
    """Answer questions about MacroMicro internal Standard Operating Procedures (SOP).

    Uses FileSearch to retrieve relevant information from the SOP documentation
    and provides detailed answers to help team members understand workflows and procedures.

    Args:
        message: The question about SOP procedures
        history: Chat history (unused - this is a memoryless service), defaults to None

    Yields:
        Detailed answer based on retrieved SOP documentation
    """
    # Memoryless service - only use the current question
    # Stream the response for better UX
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=message,
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
                yield accumulated_text
    except Exception as e:
        print(f"Error during streaming: {e}")
        if not has_yielded:
            yield "Error generating response. Please try again."
        else:
            yield accumulated_text + "\n\n[Error occurred during generation]"

    # Ensure we always yield at least once
    if not has_yielded:
        yield "No response generated. Please try rephrasing your question."


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