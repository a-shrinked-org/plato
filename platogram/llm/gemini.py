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