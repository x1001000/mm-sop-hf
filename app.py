import dotenv
dotenv.load_dotenv()

import gradio as gr

from google import genai
from google.genai import types
client = genai.Client()

def answer(
    question,
    history: list[dict[str, str]],
):
    """Answer questions about MacroMicro internal Standard Operating Procedures (SOP).

    Uses FileSearch to retrieve relevant information from the SOP documentation
    and provides detailed answers to help team members understand workflows and procedures.

    Args:
        question: The question about SOP procedures
        history: Conversation history (note: service is memoryless, only current question is used)

    Yields:
        Detailed answer based on retrieved SOP documentation
    """
    # Memoryless service - only use the current question
    # Stream the response for better UX
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction="你的任務：依據FileSearch工具檢索到的資料，詳細回答MacroMicro團隊內部標準作業流程（SOP）相關問題",
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[client.file_search_stores.list()[0].name]
                    )
                )
            ]
        )
    )

    # Stream response chunks as they arrive
    accumulated_text = ""
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
            yield accumulated_text


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