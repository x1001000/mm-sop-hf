import dotenv
dotenv.load_dotenv()

import gradio as gr

from google import genai
from google.genai import types
client = genai.Client()

# Get the file search store
file_search_stores = client.file_search_stores.list()
if not file_search_stores:
    raise ValueError("No file search stores found. Please create one in the Google AI Studio.")
file_search_store = file_search_stores[0]

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


def answer(message: str, history: list):
    """Answer questions about MacroMicro internal Standard Operating Procedures (SOP).

    Uses FileSearch to retrieve relevant information from the SOP documentation
    and provides detailed answers to help team members understand workflows and procedures.

    Args:
        message: The current input message from the user.
        history: Chat history in Gradio format.

    Yields:
        A stream of strings with the answer.
    """
    # Convert Gradio messages format to Gemini API format
    gemini_contents = []
    for user_msg, ai_msg in history:
        gemini_contents.append({"role": "user", "parts": [{"text": user_msg}]})
        gemini_contents.append({"role": "model", "parts": [{"text": ai_msg}]})
    gemini_contents.append({"role": "user", "parts": [{"text": message}]})

    # Stream the response for better UX
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

    # Stream response chunks as they arrive
    for chunk in response_stream:
        try:
            if chunk.text:
                yield chunk.text
        except ValueError:
            # This error is expected if the chunk contains a function call instead of text.
            # The Gemini API handles the tool call automatically; we just need to ignore this chunk.
            print("Ignoring chunk with function call.")
            continue


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
with gr.Blocks(fill_height=True, title="MM SOP") as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        chat_history.append((message, ""))
        for chunk in answer(message, chat_history):
            chat_history[-1] = (message, chat_history[-1][1] + chunk)
            yield "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(mcp_server=True)