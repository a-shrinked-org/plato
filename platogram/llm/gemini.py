import os
from typing import Any, Generator, Literal, Sequence
import google.generativeai as genai
from platogram.ops import render
from platogram.types import Assistant, Content, User

class Model:
    def __init__(self, model: str, key: str | None = None) -> None:
        if key is None:
            key = os.environ["GOOGLE_API_KEY"]

        # Configure the Gemini client
        genai.configure(api_key=key)

        if model == "gemini-2-pro":
            self.model = "gemini-2.0-pro-exp-02-05"
        elif model == "gemini-2-flash":
            self.model = "gemini-2.0-flash-exp"
        else:
            raise ValueError(f"Unknown model: {model}")

        self.client = genai.GenerativeModel(model_name=self.model)
        
    def get_meta(
        self,
        paragraphs: list[str],
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> tuple[str, str]:
        if not lang:
            lang = "en"
    
        # Similar system prompts as anthropic.py but adjusted for Gemini
        system_prompt = {
            "en": """You are an editor tasked with extracting a title and summary from content.
    Use simple language and only words from the provided text.""",
            "es": """Eres un editor encargado de extraer un título y resumen del contenido.
    Utiliza un lenguaje sencillo y solo palabras del texto proporcionado."""
        }
    
        text = "\n".join([f"<p>{paragraph}</p>" for paragraph in paragraphs])
        
        # Define tool schema similar to anthropic.py
        tool_definition = {
            "name": "render_content_info",
            "description": "Renders useful information about text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"}
                },
                "required": ["title", "summary"]
            }
        }
    
        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<text>{text}</text>")],
            tools=[tool_definition],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
        # Parse response and return title, summary
        if isinstance(response, dict):
            return response["title"], response["summary"]
        raise ValueError(f"Expected dict response, got {type(response)}")

    def count_tokens(self, text: str) -> int:
        """Count tokens for a given text using Gemini's API"""
        return self.client.count_tokens(text).total_tokens

    def prompt_model(
        self,
        messages: Sequence[User | Assistant],
        max_tokens: int = 4096,
        temperature=0.1,
        stream=False,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> str | dict[str, str] | Generator[str, None, None]:
        # Convert messages to Gemini format
        gemini_messages = []
        if system:
            gemini_messages.append({"role": "system", "content": system})
        
        for m in messages:
            gemini_messages.append({
                "role": "user" if isinstance(m, User) else "assistant",
                "content": m.content
            })

        response = self.client.generate_content(
            gemini_messages,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            },
            tools=tools if tools else None,
            stream=stream
        )

        if stream:
            def stream_text():
                for chunk in response:
                    yield chunk.text
            return stream_text()

        return response.text