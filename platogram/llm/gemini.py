import os
import re
import time
import logging
from typing import Any, Generator, Literal, Sequence

from google import genai
from google.genai import types
from google.oauth2 import service_account

from platogram.ops import render
from platogram.types import Assistant, Content, User

class Model:
    def __init__(self, model: str = "gemini-2.0-flash-001", key: str | None = None):
        print("Debug: Initializing Gemini with project:", os.getenv('GOOGLE_CLOUD_PROJECT'))
        print("Debug: Using credentials from:", os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
        
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location="us-central1",
            credentials=credentials,
            http_options=types.HttpOptions(api_version='v1')
        )
        
        self.model_name = model
        print(f"Debug: Using model: {model}")

    def prompt_model(
        self,
        messages: Sequence[User | Assistant],
        max_tokens: int = 4096,
        temperature: float = 0.1,
        stream: bool = False,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> str | dict[str, str] | Generator[str, None, None]:
        logger = logging.getLogger('gemini')
        try:
            logger.info(f"Starting request with model: {self.model_name}")
            
            contents = []
            if system:
                contents.append(types.Content(
                    parts=[types.Part.from_text(text=system)],
                    role="user"
                ))
            
            for m in messages:
                if isinstance(m, (User, Assistant)):
                    role = "user" if isinstance(m, User) else "model"
                    content = m.content
                elif isinstance(m, dict):
                    role = "user" if m.get("role", "") == "user" else "model"
                    content = m.get("content", "")
                else:
                    raise ValueError(f"Unsupported message type: {type(m)}")
                
                contents.append(types.Content(
                    parts=[types.Part.from_text(text=str(content))],
                    role=role
                ))
    
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40,
            )
    
            # Removed JSON instruction since we expect Markdown output with flags
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
    
            response_text = response.text.strip()
            return response_text  # Return Markdown directly
            
        except Exception as e:
            logger.error(f"Error in prompt_model: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        try:
            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=text
            )
            return response.total_tokens
        except Exception as e:
            return len(text.split()) * 2

    def get_meta(
        self,
        paragraphs: list[str],
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> tuple[str, str]:
        if not lang:
            lang = "en"
    
        system_prompt = {
            "en": """You are a skilled academic editor tasked with analyzing content to produce a clear title and comprehensive summary.
Given the text, generate a concise title (1-2 lines) capturing the essence and a comprehensive summary (3-5 sentences) covering all key points, using only words from the text.
Output in Markdown: `# Title\n\n## Abstract\n\nSummary`""".strip(),
            "es": """Eres un editor académico experto encargado de analizar contenido para producir un título claro y un resumen completo.
Dado el texto, genera un título conciso (1-2 líneas) que capte la esencia y un resumen completo (3-5 oraciones) que cubra todos los puntos clave, usando solo palabras del texto.
Salida en Markdown: `# Título\n\n## Resumen\n\nResumen`""".strip(),
        }
    
        text = "\n".join([f"<p>{paragraph}</p>" for paragraph in paragraphs])
        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=text)],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
        title_match = re.search(r'# (.*?)\n', response, re.DOTALL)
        summary_match = re.search(r'## (?:Abstract|Resumen)\n\n(.*?)$', response, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Generated Document"
        summary = summary_match.group(1).strip() if summary_match else ""
        return title, summary
    
    def get_chapters(
        self,
        passages: list[str],
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> dict[int, str]:
        if not lang:
            lang = "en"
    
        system_prompt = {
            "en": """You are a skilled academic editor organizing content into logical chapters.
Analyze the text, identify major themes, and create logical chapter divisions with timestamps.
Output in Markdown with each chapter as `**[timestamp] Title**` on a new line.""".strip(),
            "es": """Eres un editor académico experto organizando contenido en capítulos lógicos.
Analiza el texto, identifica temas principales y crea divisiones de capítulos lógicos con marcas de tiempo.
Salida en Markdown con cada capítulo como `**[marca de tiempo] Título**` en una nueva línea.""".strip(),
        }
    
        text = "\n".join([f"<p>{passage}</p>" for passage in passages])
        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=text)],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
        chapters = {}
        for line in response.split('\n'):
            match = re.search(r'\*\*\[(.*?)\]\s+(.*?)\*\*', line)
            if match:
                timestamp = match.group(1)
                title = match.group(2).strip()
                chapters[timestamp] = title
        return chapters
    
    def get_paragraphs(
        self,
        text_with_markers: str,
        examples: dict[str, list[str]],
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> list[str]:
        if not lang:
            lang = "en"
    
        system_prompt = {
            "en": """You are a skilled academic writer transforming text into well-structured paragraphs.
Transform the text into clear, structured paragraphs, preserving all markers.
Output in Markdown with each paragraph enclosed in `<p>...</p>` tags.""".strip(),
            "es": """Eres un escritor académico experto transformando texto en párrafos bien estructurados.
Transforma el texto en párrafos claros y estructurados, preservando todos los marcadores.
Salida en Markdown con cada párrafo encerrado en etiquetas `<p>...</p>`.""".strip(),
        }
    
        messages = []
        for prompt, paragraphs in examples.items():
            formatted_response = "\n".join([f"<p>{p}</p>" for p in paragraphs])
            messages.extend([
                {"role": "user", "content": f"Transform this text:\n{prompt}"},
                {"role": "assistant", "content": formatted_response}
            ])
        messages.append({"role": "user", "content": f"Transform this text:\n{text_with_markers}"})
    
        response = self.prompt_model(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt[lang]
        )
        return re.findall(r"<p>(.*?)</p>", response, re.DOTALL)
    
    def render_context(
        self, context: list[Content], context_size: Literal["small", "medium", "large"]
    ) -> str:
        base = 0
        output = ""
    
        for content in context:
            paragraphs = [
                re.sub(r"【(\d+)】", lambda m: f"【{int(m.group(1))+base}】", paragraph)
                for paragraph in content.passages
            ]
            paragraphs = [
                re.sub(
                    r"【(\d+)】(\w*【\d+】\w*)+",
                    lambda m: f"【{int(m.group(1))}】",
                    paragraph,
                )
                for paragraph in paragraphs
            ]
            output += f'<content title="{content.title}" summary="{content.summary}">\n'
            if context_size == "small" or context_size == "large":
                output += "<paragraphs>\n"
                output += "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
                output += "\n</paragraphs>\n"
    
            if context_size == "medium" or context_size == "large":
                text = render(
                    {i + base: event.text for i, event in enumerate(content.transcript)}
                )
                output += f"<text>{text}</text>\n"
    
            output += "</content>\n"
            base += len(content.transcript)
    
        return output.strip()
    
    def prompt(
        self,
        prompt: Sequence[User | Assistant] | str,
        *,
        context: list[Content],
        context_size: Literal["small", "medium", "large"] = "small",
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> str:
        if not lang:
            lang = "en"
    
        system_prompt = {
            "en": """You are a skilled academic researcher analyzing content and providing well-structured responses.
Analyze the context and prompt, then generate a response in Markdown format.""".strip(),
            "es": """Eres un investigador académico experto analizando contenido y proporcionando respuestas bien estructuradas.
Analiza el contexto y el prompt, luego genera una respuesta en formato Markdown.""".strip(),
        }
    
        if isinstance(prompt, str):
            prompt = [User(content=prompt)]
    
        response = self.prompt_model(
            max_tokens=max_tokens,
            messages=[
                User(
                    content=f"""Context:\n{self.render_context(context, context_size)}\n\nQuery:\n{prompt}"""
                )
            ],
            system=system_prompt[lang],
            temperature=temperature,
        )
        return response

    def get_contributors(
        self,
        text: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> list[tuple[str, str, str]]:
        if not lang:
            lang = "en"

        system_prompt = {
            "en": """You are a skilled editor extracting contributor information.
Analyze the text and extract names, roles, and organizations in Markdown list format: `- Name, Role, Organization`.
Use "Unknown" for missing info.""".strip(),
            "es": """Eres un editor experto extrayendo información de contribuyentes.
Analiza el texto y extrae nombres, roles y organizaciones en formato de lista Markdown: `- Nombre, Rol, Organización`.
Usa "Desconocido" para información faltante.""".strip(),
        }

        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<text>{text}</text>")],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        contributors = []
        for line in response.split('\n'):
            if line.startswith('- '):
                parts = line[2:].split(', ')
                if len(parts) == 3:
                    contributors.append((parts[0], parts[1], parts[2]))
        return contributors

    def get_conclusion(
        self,
        text: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> str:
        if not lang:
            lang = "en"

        system_prompt = {
            "en": """You are a skilled academic editor creating conclusions.
Analyze the text and generate a conclusion in Markdown format with 4-5 paragraphs, including marker references [number].""".strip(),
            "es": """Eres un editor académico experto creando conclusiones.
Analiza el texto y genera una conclusión en formato Markdown con 4-5 párrafos, incluyendo referencias de marcadores [número].""".strip(),
        }

        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<text>{text}</text>")],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.strip()