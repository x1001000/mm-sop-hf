import dotenv
dotenv.load_dotenv()

import gradio as gr

from google import genai
from google.genai import types
client = genai.Client()

def respond(
    message,
    history: list[dict[str, str]],
):
    user_message = ''
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        user_message += f"{role}: {content}\n"
    user_message += f"user: {message}"

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=user_message,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[client.file_search_stores.list()[0].name]
                    )
                )
            ]
        )
    )

    # Extract and return the response text
    result = response.text if hasattr(response, 'text') else str(response)
    yield result


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
)

with gr.Blocks() as demo:
    chatbot.render()

if __name__ == "__main__":
    demo.launch(mcp_server=True)