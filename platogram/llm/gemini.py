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
            "en": """<role>
You are a very capable editor, speaker, educator, and author with a knack for coming up with information about the content that helps to understand it.
</role>
<task>
You will be given a <text> that contains paragraphs enclosed in <p></p> tags and you will need to come up with information about the content that helps to understand it.
Use simple language. Use only the words from <text>.
Always call render_content_info tool and pass the information about the content.
</task>""".strip(),
            "es": """<role>
Eres un editor, orador, educador y autor muy capaz con un don para crear información sobre el contenido que ayuda a entenderlo.
</role>
<task>
Se te dará un <text> que contiene párrafos encerrados en etiquetas <p></p> y tendrás que crear información sobre el contenido que ayude a entenderlo.
Utiliza un lenguaje sencillo. Usa solo las palabras de <text>.
Siempre llama al tool render_content_info y pasa la información sobre el contenido.
</task>""".strip(),
        }
    
        text = "\n".join([f"<p>{paragraph}</p>" for paragraph in paragraphs])
        
        tool_definition = {
            "name": "render_content_info",
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
    
        if isinstance(response, dict):
            return response.get("title", ""), response.get("summary", "")
            
        raise ValueError(f"Expected dict response, got {type(response)}")
    
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
            "en": """<role>
You are a very capable editor, speaker, educator, and author who is really good at reading text that represents transcript of human speech and rewriting it into well-structured, information-dense written text.
</role>
<task>
You will be given <passages> of text in a format "<p>text【0】text【1】... text 【2】</p>". Where each【number】is a <marker> and goes AFTER the text it references. Markers are zero-based and are in sequential order.

Follow these steps to transform the <passages> into a dictionary of chapters:
1. Read the <passages> carefully and come up with a list of <chapters> that equally cover the <passages>.
2. Review <chapters> and <passages> and for each chapter generate <title> and first <marker> from the relevant passage and create a <chapter> object and add it to the list.
3. Call <chapter_tool> and pass the list of <chapter> objects.
</task>""".strip(),
            "es": """Eres un editor, orador, educador y autor muy capaz con un don para crear información sobre el contenido que ayuda a entenderlo.
    
    Tarea:
    Se te dará un <text> que contiene párrafos encerrados en etiquetas <p></p> y tendrás que crear información sobre el contenido que ayude a entenderlo.
    Utiliza un lenguaje sencillo. Usa solo las palabras de <text>.
    Siempre llama al tool render_content_info y pasa la información sobre el contenido.
</task>""".strip(),
        }
    
        text = "\n".join([f"<p>{passage}</p>" for passage in passages])
        
        tool_definition = {
            "name": "chapter_tool",
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
    
        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<passages>{text}</passages>")],
            tools=[tool_definition],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
        if isinstance(response, dict) and "entities" in response:
            return {
                int(re.findall(r"\d+", chapter["marker"])[0]): chapter["title"].strip()
                for chapter in response["entities"]
            }
            
        raise ValueError(f"Expected dict with entities, got {type(response)}")
    
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
        try:
            return self.client.count_tokens(text).total_tokens
        except Exception as e:
            # Fallback to rough estimation if API fails
            return len(text.split()) * 2  # Rough approximation

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
            # Handle both text and structured content
            content = m.content
            if hasattr(m, 'cache') and m.cache:
                # Handle cached content similar to Anthropic's ephemeral caching
                content = {"text": m.content, "cache_control": "ephemeral"}
            
            gemini_messages.append({
                "role": "user" if isinstance(m, User) else "assistant",
                "content": content
            })

        max_retries = 3
        backoff = 2

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    gemini_messages,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "candidate_count": 1,
                    },
                    tools=tools if tools else None,
                    stream=stream
                )

                if stream:
                    def stream_text():
                        for chunk in response:
                            if hasattr(chunk, 'text'):
                                yield chunk.text
                    return stream_text()

                # Handle function calling/tools response
                if tools and hasattr(response, 'candidates') and response.candidates[0].content.parts[0].function_call:
                    function_call = response.candidates[0].content.parts[0].function_call
                    return function_call.args
                
                # Handle regular text response
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    return response.candidates[0].content.text

                raise ValueError(f"Unexpected response format: {response}")

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff ** attempt)
                continue