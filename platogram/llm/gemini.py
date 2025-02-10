import os
import re

import time 

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
    
        system_prompt = {
            "en": """You are an editor tasked with extracting a title and summary from content.
    Use simple language and only words from the provided text.
    Always use the render_content_info function to return your response.""",
            "es": """Eres un editor encargado de extraer un título y resumen del contenido.
    Utiliza un lenguaje sencillo y solo palabras del texto proporcionado.
    Siempre usa la función render_content_info para devolver tu respuesta."""
        }
    
        text = "\n".join([f"<p>{paragraph}</p>" for paragraph in paragraphs])
        
        tool_definition = {
            "function": "render_content_info",
            "description": "Renders useful information about text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title that captures the essence of the content"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Comprehensive summary of the content"
                    }
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
    
        # Handle Gemini's function call response format
        if hasattr(response, 'candidates') and response.candidates[0].content.parts[0].function_call:
            chapters = response.candidates[0].content.parts[0].function_call.args
        elif isinstance(response, dict):
            chapters = response
        else:
            raise ValueError(f"Expected function call or dict response, got {type(response)}")
        
        assert isinstance(chapters.get("entities"), list), f"Expected list, got {type(chapters.get('entities'))}"
        
        return {
            int(re.findall(r"\d+", chapter["marker"])[0]): chapter["title"].strip()
            for chapter in chapters["entities"]
        }
        
        # Fallback for direct dictionary response
        if isinstance(response, dict):
            return response.get("title"), response.get("summary")
            
        raise ValueError(f"Expected function call or dict response, got {type(response)}")
    
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
            "en": """You are a skilled academic editor transforming text into well-organized chapters.
    
    Task:
    You will receive <passages> containing text with markers【number】. Each marker goes AFTER its reference text and is zero-based and sequential.
    
    Transform these passages into well-structured chapters by:
    1. Carefully analyzing all passages to identify main topics
    2. Creating logical chapter divisions that cover all content
    3. Generating clear, descriptive titles for each chapter
    4. Identifying the first marker for each chapter section
    5. Maintaining academic style and formal tone
    
    Expected output: A list of chapters with titles and their starting markers.""",
            "es": """Eres un editor académico experto transformando texto en capítulos bien organizados.
    
    Tarea:
    Recibirás <passages> conteniendo texto con marcadores【número】. Cada marcador va DESPUÉS de su texto de referencia y está basado en cero y es secuencial.
    
    Transforma estos pasajes en capítulos bien estructurados:
    1. Analizando cuidadosamente todos los pasajes para identificar temas principales
    2. Creando divisiones lógicas de capítulos que cubran todo el contenido
    3. Generando títulos claros y descriptivos para cada capítulo
    4. Identificando el primer marcador para cada sección de capítulo
    5. Manteniendo estilo académico y tono formal
    
    Salida esperada: Una lista de capítulos con títulos y sus marcadores iniciales."""
        }
    
        tool_definition = {
            "function": "chapter_tool", 
            "description": "Creates chapters from passages",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Title of the chapter"
                                },
                                "marker": {
                                    "type": "string",
                                    "description": "First marker in format '【number】'"
                                }
                            },
                            "required": ["title", "marker"]
                        }
                    }
                },
                "required": ["entities"]
            }
        }
    
        text = "\n".join([f"<p>{passage}</p>" for passage in passages])
        chapters = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<passages>{text}</passages>")],
            tools=[tool_definition],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
        assert isinstance(chapters, dict), f"Expected dict, got {type(chapters)}"
        assert isinstance(chapters["entities"], list), f"Expected list, got {type(chapters['entities'])}"
        
        return {
            int(re.findall(r"\d+", chapter["marker"])[0]): chapter["title"].strip()
            for chapter in chapters["entities"]
        }
    
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
            "en": """You are a skilled academic writer transforming informal speech into well-structured, formal academic prose.
    
    Style Requirements:
    - Clear topic sentences introducing main ideas
    - Well-organized supporting details
    - Proper paragraph structure and flow
    - Natural transitions between ideas
    - Consistent academic tone
    - Natural integration of all markers
    
    Task Details:
    You will receive text with numeric markers【number】that appear AFTER their reference text.
    Transform this text while:
    1. Converting conversational speech to formal academic prose
    2. Structuring content with clear topic sentences
    3. Using appropriate transitions
    4. Maintaining consistent academic style
    5. Preserving all markers in exact sequence
    6. Integrating markers naturally into text
    7. Aiming for publication-ready quality
    
    Study the provided examples carefully and match their style.""",
            "es": """Eres un escritor académico experto que transforma el habla informal en prosa académica formal.
    
    Requisitos de Estilo:
    - Oraciones temáticas claras que introducen ideas principales
    - Detalles de apoyo bien organizados
    - Estructura y flujo apropiado de párrafos
    - Transiciones naturales entre ideas
    - Tono académico consistente
    - Integración natural de todos los marcadores
    
    Detalles de la Tarea:
    Recibirás texto con marcadores numéricos【número】que aparecen DESPUÉS de su texto de referencia.
    Transforma este texto mientras:
    1. Conviertes el habla conversacional en prosa académica formal
    2. Estructuras el contenido con oraciones temáticas claras
    3. Usas transiciones apropiadas
    4. Mantienes un estilo académico consistente
    5. Preservas todos los marcadores en secuencia exacta
    6. Integras los marcadores naturalmente en el texto
    7. Buscas una calidad lista para publicación
    
    Estudia los ejemplos proporcionados cuidadosamente y coincide con su estilo."""
        }
    
        # Format examples
        messages = []
        for prompt, paragraphs in examples.items():
            messages.extend([
                {"role": "user", "content": f"<transcript>{prompt}</transcript>"},
                {"role": "assistant", "content": f"<paragraphs>\n" + "\n".join([f"<p>{p}</p>" for p in paragraphs]) + "\n</paragraphs>"}
            ])
    
        messages.extend([
            {"role": "user", "content": f"<transcript>{text_with_markers}</transcript>"},
            {"role": "assistant", "content": "<paragraphs>\n<p>"}
        ])
    
        response = self.prompt_model(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt[lang]
        )
    
        assert isinstance(response, str)
        return re.findall(r"<p>(.*?)</p>", response, re.DOTALL)
    
    def render_context(
        self,
        context: list[Content],
        context_size: Literal["small", "medium", "large"]
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
            "en": """You are a skilled academic researcher creating information-dense responses supported by context references.
    
    Task:
    Process the given <context> and <prompt> where:
    - Each <content> contains information with title and summary
    - Content has <text> (speech transcript) and/or <paragraphs> (structured text)
    - Special markers【number】appear AFTER referenced text
    - Multiple consecutive markers are treated as one reference point
    
    Your response should:
    1. Be well-structured and information-dense
    2. Cover all relevant parts of prompt and context
    3. Include ALL supporting markers from context
    4. Maintain academic style and formal tone
    5. Use natural transitions and flow
    6. Integrate markers seamlessly into text""",
            "es": """Eres un investigador académico experto creando respuestas densas en información respaldadas por referencias del contexto.
    
    Tarea:
    Procesa el <context> y <prompt> dados donde:
    - Cada <content> contiene información con título y resumen
    - El contenido tiene <text> (transcripción) y/o <paragraphs> (texto estructurado)
    - Marcadores especiales【número】aparecen DESPUÉS del texto referenciado
    - Múltiples marcadores consecutivos se tratan como un punto de referencia
    
    Tu respuesta debe:
    1. Estar bien estructurada y densa en información
    2. Cubrir todas las partes relevantes del prompt y contexto
    3. Incluir TODOS los marcadores de apoyo del contexto
    4. Mantener estilo académico y tono formal
    5. Usar transiciones naturales y flujo
    6. Integrar marcadores perfectamente en el texto"""
        }
    
        if isinstance(prompt, str):
            prompt = [User(content=prompt)]
    
        response = self.prompt_model(
            max_tokens=max_tokens,
            messages=[
                User(
                    content=f"""<context>
    {self.render_context(context, context_size)}
    </context>
    <prompt>
    {prompt}
    </prompt>"""
                )
            ],
            system=system_prompt[lang],
            temperature=temperature,
        )
        
        assert isinstance(response, str)
        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens for a given text using Gemini's API"""
        return self.client.count_tokens(text).total_tokens

    # Add this error handling to prompt_model method:
    
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
    
        max_retries = 3
        backoff = 2
    
        for attempt in range(max_retries):
            try:
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
    
                if tools and hasattr(response, 'candidates') and response.candidates[0].content.parts[0].function_call:
                    return response.candidates[0].content.parts[0].function_call.args
                
                return response.text
    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff ** attempt)
                continue