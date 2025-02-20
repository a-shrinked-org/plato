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
        
        # Load credentials with proper scopes
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Create client for Vertex AI
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
            
            # Convert messages to content format
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

            # Configure generation
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40,
            )

            # Handle tools if provided
            if tools:
                function_declarations = []
                for tool in tools:
                    if "input_schema" in tool:
                        function_declarations.append(types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=tool["input_schema"]
                        ))
                    else:
                        function_declarations.append(tool)
                config.tools = [types.Tool(function_declarations=function_declarations)]

            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            # Handle function calls in response
            if response.candidates and response.candidates[0].content.parts[0].function_call:
                return {
                    k: v for k, v in response.candidates[0].content.parts[0].function_call.args.items()
                }
            
            return response.text.strip()

        except Exception as e:
            logger.error(f"Error in prompt_model: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens for a given text"""
        try:
            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=text
            )
            return response.total_tokens
        except Exception as e:
            # Fallback to rough estimation
            return len(text.split()) * 2

    # (get_meta, get_chapters, get_paragraphs, prompt, render_context)
        
    # Update to gemini.py
    
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
    You are a skilled academic editor analyzing content.
    Your task is to extract a clear title and comprehensive summary.
    </role>
    <task>
    1. Study the provided text carefully
    2. Extract a clear title and comprehensive summary that represents the full scope
    3. Use only information present in the text
    4. Output in a clean format without newlines or special characters
    5. Keep titles concise (1-2 lines) and summaries comprehensive but focused
    </task>""",
            "es": """<role>
    Eres un editor académico experto analizando contenido.
    Tu tarea es extraer un título claro y un resumen completo.
    </role>
    <task>
    1. Estudia el texto proporcionado cuidadosamente
    2. Extrae un título claro y un resumen completo que represente todo el alcance
    3. Usa solo información presente en el texto
    4. Produce formato limpio sin saltos de línea o caracteres especiales
    5. Mantén títulos concisos (1-2 líneas) y resúmenes completos pero enfocados
    </task>"""
        }
    
        # Update tool definition to match Anthropic's schema
        tool_definition = {
            "name": "render_content_info",
            "description": "Creates title and summary from content",
            "input_schema": {  # Changed from parameters to input_schema
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Concise, descriptive title without newlines"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Comprehensive summary without newlines"
                    }
                },
                "required": ["title", "summary"]
            }
        }
    
        text = "\n".join([f"<p>{paragraph}</p>" for paragraph in paragraphs])
    
        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<text>{text}</text>")],
            tools=[tool_definition],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
        if isinstance(response, dict):
            # Clean and validate the response
            title = response.get("title", "").replace("\n", " ").strip()
            summary = response.get("summary", "").replace("\n", " ").strip()
            
            if not title or not summary:
                raise ValueError("Missing title or summary in response")
                
            # Clean any potential control characters
            title = "".join(char for char in title if char.isprintable())
            summary = "".join(char for char in summary if char.isprintable())
            
            return title, summary
            
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
            "en": """You are a skilled academic editor organizing content into logical chapters.
    
    Instructions:
    1. Analyze the full text carefully
    2. Identify major themes and topics
    3. Create logical chapter divisions
    4. Preserve all numerical markers
    5. Maintain academic style
    
    Example:
    Input: Text with markers about quantum computing:
    "Basic principles of quantum states【0】. Qubits represent... Quantum gates enable operations【1】"
    Output: {
        "entities": [
            {"title": "Quantum State Fundamentals", "marker": "【0】"},
            {"title": "Quantum Gate Operations", "marker": "【1】"}
        ]
    }
    
    Format Requirements:
    - Each chapter must have a clear, descriptive title
    - Titles should be 3-7 words
    - Preserve exact marker format: 【number】
    - Markers must appear in sequence
    - Maintain academic tone throughout""",
    
            "es": """Eres un editor académico experto organizando contenido en capítulos lógicos.
    
    Instrucciones:
    1. Analiza el texto completo cuidadosamente
    2. Identifica temas y tópicos principales
    3. Crea divisiones lógicas de capítulos
    4. Preserva todos los marcadores numéricos
    5. Mantén estilo académico
    
    Ejemplo:
    Entrada: Texto con marcadores sobre computación cuántica:
    "Principios básicos de estados cuánticos【0】. Los qubits representan... Las puertas cuánticas permiten operaciones【1】"
    Salida: {
        "entities": [
            {"title": "Fundamentos de Estados Cuánticos", "marker": "【0】"},
            {"title": "Operaciones de Puertas Cuánticas", "marker": "【1】"}
        ]
    }
    
    Requisitos de Formato:
    - Cada capítulo debe tener un título claro y descriptivo
    - Títulos deben ser 3-7 palabras
    - Preservar formato exacto de marcadores: 【número】
    - Marcadores deben aparecer en secuencia
    - Mantener tono académico"""
        }
    
        text = "\n".join([f"<p>{passage}</p>" for passage in passages])
        
        tool_definition = {
            "name": "chapter_tool",
            "description": "Creates structured chapters from content",
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
                                    "description": "Chapter title"
                                },
                                "marker": {
                                    "type": "string",
                                    "description": "Starting marker in format 【number】"
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
            "en": """You are a skilled academic writer transforming text into well-structured paragraphs.
    
    Instructions:
    1. Study the provided examples carefully
    2. Transform text while preserving all markers
    3. Create clear topic sentences
    4. Use appropriate transitions
    5. Maintain consistent academic style
    
    Example Format:
    Input: "Quantum computing uses quantum bits【0】. These qubits allow... Superposition enables parallel processing【1】"
    Output: "<p>The fundamental principle of quantum computing relies on quantum bits or qubits【0】. Through the phenomenon of superposition, these quantum systems enable unprecedented parallel processing capabilities【1】.</p>"
    
    Format Requirements:
    - Each paragraph must have a clear topic sentence
    - Use appropriate transitions between ideas
    - Preserve exact marker format: 【number】
    - Maintain markers in sequence
    - Use formal academic tone
    - Enclose each paragraph in <p>...</p> tags""",
    
            "es": """Eres un escritor académico experto transformando texto en párrafos bien estructurados.
    
    Instrucciones:
    1. Estudia los ejemplos proporcionados cuidadosamente
    2. Transforma el texto preservando todos los marcadores
    3. Crea oraciones temáticas claras
    4. Usa transiciones apropiadas
    5. Mantén estilo académico consistente
    
    Formato de Ejemplo:
    Entrada: "La computación cuántica usa bits cuánticos【0】. Estos qubits permiten... La superposición permite procesamiento paralelo【1】"
    Salida: "<p>El principio fundamental de la computación cuántica se basa en bits cuánticos o qubits【0】. A través del fenómeno de superposición, estos sistemas cuánticos permiten capacidades de procesamiento paralelo sin precedentes【1】.</p>"
    
    Requisitos de Formato:
    - Cada párrafo debe tener una oración temática clara
    - Usar transiciones apropiadas entre ideas
    - Preservar formato exacto de marcadores: 【número】
    - Mantener marcadores en secuencia
    - Usar tono académico formal
    - Encerrar cada párrafo en etiquetas <p>...</p>"""
        }
    
        messages = []
        # Format examples
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
    
        assert isinstance(response, str)
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
    
    Instructions:
    1. Analyze all provided content carefully
    2. Incorporate context appropriately
    3. Structure response logically
    4. Preserve all numerical markers
    5. Maintain academic style
    
    Example:
    Content: "Quantum theory introduction【0】. Wave-particle duality【1】"
    Query: "Explain quantum mechanics basics"
    Response: "The foundations of quantum mechanics begin with its core principles【0】. A central concept is wave-particle duality, which demonstrates the unique behavior of quantum systems【1】."
    
    Format Requirements:
    - Use clear topic sentences
    - Include appropriate transitions
    - Preserve exact marker format: 【number】
    - Maintain markers in sequence
    - Use formal academic tone
    - Reference context accurately""",
    
            "es": """Eres un investigador académico experto analizando contenido y proporcionando respuestas bien estructuradas.
    
    Instrucciones:
    1. Analiza todo el contenido proporcionado cuidadosamente
    2. Incorpora contexto apropiadamente
    3. Estructura la respuesta lógicamente
    4. Preserva todos los marcadores numéricos
    5. Mantén estilo académico
    
    Ejemplo:
    Contenido: "Introducción a la teoría cuántica【0】. Dualidad onda-partícula【1】"
    Consulta: "Explica los fundamentos de la mecánica cuántica"
    Respuesta: "Los fundamentos de la mecánica cuántica comienzan con sus principios básicos【0】. Un concepto central es la dualidad onda-partícula, que demuestra el comportamiento único de los sistemas cuánticos【1】."
    
    Requisitos de Formato:
    - Usar oraciones temáticas claras
    - Incluir transiciones apropiadas
    - Preservar formato exacto de marcadores: 【número】
    - Mantener marcadores en secuencia
    - Usar tono académico formal
    - Referenciar contexto con precisión"""
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
        
        assert isinstance(response, str)
        return response

    def get_contributors(
        self,
        text: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Extract contributors from content."""
        logger = logging.getLogger('gemini')
        if not lang:
            lang = "en"

        system_prompt = {
            "en": """You are a skilled editor extracting contributor information.

Instructions:
1. Analyze the text carefully
2. Identify all mentioned individuals
3. Extract their names, roles, and organizations
4. Only include explicitly stated information
5. Use "Unknown" for missing information

Format Requirements:
- Name: Full name if available
- Role: Professional title or role
- Organization: Company or institution
- Only include information present in text
- Do not make assumptions""",
            "es": """Eres un editor experto extrayendo información de contribuyentes.

Instructions:
1. Analiza el texto cuidadosamente
2. Identifica todas las personas mencionadas
3. Extrae sus nombres, roles y organizaciones
4. Solo incluye información explícitamente mencionada
5. Usa "Desconocido" para información faltante

Requisitos de Formato:
- Nombre: Nombre completo si está disponible
- Rol: Título profesional o rol
- Organización: Compañía o institución
- Solo incluir información presente en el texto
- No hacer suposiciones"""
        }

        tool_definition = {
            "name": "extract_contributors",
            "description": "Extracts contributor information from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "contributors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role": {"type": "string"},
                                "organization": {"type": "string"}
                            },
                            "required": ["name", "role", "organization"]
                        }
                    }
                },
                "required": ["contributors"]
            }
        }

        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<text>{text}</text>")],
            tools=[tool_definition],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if isinstance(response, dict) and "contributors" in response:
            return [(c["name"], c["role"], c["organization"]) 
                    for c in response["contributors"]]
        
        raise ValueError(f"Expected dict with contributors, got {response}")

    def get_conclusion(
        self,
        text: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        lang: str | None = None,
    ) -> str:
        """Generate conclusion from content."""
        logger = logging.getLogger('gemini')
        if not lang:
            lang = "en"

        system_prompt = {
            "en": """You are a skilled academic editor creating conclusions.

Instructions:
1. Analyze the full text carefully
2. Identify key findings and implications
3. Summarize main points and their significance
4. Maintain academic style
5. Include relevant numerical markers

Format Requirements:
- 4-5 paragraphs
- Include marker references [number]
- Maintain formal tone
- Focus on key insights
- End with future implications""",
            "es": """Eres un editor académico experto creando conclusiones.

Instructions:
1. Analiza el texto completo cuidadosamente
2. Identifica hallazgos e implicaciones clave
3. Resume puntos principales y su significado
4. Mantén estilo académico
5. Incluye marcadores numéricos relevantes

Requisitos de Formato:
- 4-5 párrafos
- Incluir referencias de marcadores [número]
- Mantener tono formal
- Enfocarse en insights clave
- Terminar con implicaciones futuras"""
        }

        response = self.prompt_model(
            system=system_prompt[lang],
            messages=[User(content=f"<text>{text}</text>")],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if isinstance(response, str):
            return response.strip()
        
        raise ValueError(f"Expected string response, got {type(response)}")
