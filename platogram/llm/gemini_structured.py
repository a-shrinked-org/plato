# In platogram/llm/gemini_structured.py
import re
import json
from typing import List
from platogram.llm.gemini import Model
from platogram.types import SpeechEvent

class StructuredGemini(Model):
    def generate_structured_output(self, transcript: List[SpeechEvent]):
        # System prompt with detailed role and task definitions
        system_prompt = """
        <role>
        You are an academic editor analyzing a transcript to produce a structured document.
        </role>
        <task>
        1. Analyze the transcript carefully, noting timestamps in [ms] format.
        2. Generate a concise title that captures the essence of the content.
        3. Write an abstract summarizing the main points.
        4. Identify chapters with descriptive titles and their starting timestamps (e.g., 'Introduction [0ms]').
        5. Extract key passages with timestamps (e.g., '[500ms] Text here').
        6. List any references or citations mentioned in the transcript.
        7. Return a JSON object with 'title' (str), 'abstract' (str), 'chapters' (dict of timestamp: title), 'passages' (list), and 'references' (list).
        </task>
        """

        # Few-shot examples with full structure
        few_shot_examples = """
        Example 1:
        Input Transcript:
        [0ms] This is a sample transcript about quantum computing. [500ms] It explains how quantum computers use quantum bits to perform calculations faster. [1000ms] Reference: Quantum Computing for Dummies.

        Output:
        {
            "title": "Introduction to Quantum Computing",
            "abstract": "This transcript provides an overview of quantum computing, including its principles and applications.",
            "chapters": {
                "0": "Quantum Computing Basics",
                "500": "Quantum Bits and Calculations"
            },
            "passages": [
                "[0ms] This is a sample transcript about quantum computing.",
                "[500ms] It explains how quantum computers use quantum bits to perform calculations faster."
            ],
            "references": ["Quantum Computing for Dummies"]
        }

        Example 2:
        Input Transcript:
        [0ms] Discussion on AI ethics. [1000ms] The importance of unbiased algorithms. [1500ms] Reference: Ethics of Artificial Intelligence.

        Output:
        {
            "title": "Ethics in Artificial Intelligence",
            "abstract": "This transcript explores the ethical considerations in AI development.",
            "chapters": {
                "0": "Introduction to AI Ethics",
                "1000": "Algorithmic Bias"
            },
            "passages": [
                "[0ms] Discussion on AI ethics.",
                "[1000ms] The importance of unbiased algorithms."
            ],
            "references": ["Ethics of Artificial Intelligence"]
        }
        """

        # Convert transcript to string with timestamps
        transcript_text = "\n".join(f"[{event.time_ms}ms] {event.text}" for event in transcript)

        # User prompt with prefixes and partial input completion
        user_prompt = f"""
        <examples>
        {few_shot_examples}
        </examples>

        <input>
        Input Transcript:
        {transcript_text}
        </input>

        <output>
        {{
            "title": "",
            "abstract": "",
            "chapters": {{}},
            "passages": [],
            "references": []
        }}
        </output>
        """

        # Call the model with the enhanced prompt
        raw_output = self.prompt_model(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4096,
            temperature=0.2
        )
        try:
            if isinstance(raw_output, str):
                raw_output = re.sub(r'^```json\s*|\s*```$', '', raw_output, flags=re.MULTILINE).strip()
                output = json.loads(raw_output)
            else:
                output = raw_output
            return {
                "title": output.get("title", "Generated Document"),
                "abstract": output.get("abstract", "No summary available"),
                "chapters": output.get("chapters", {}),
                "passages": output.get("passages", []),
                "references": output.get("references", [])
            }
        except Exception as e:
            print(f"Error parsing Gemini output: {e}")
            return {
                "title": "Generated Document",
                "abstract": transcript_text[:200] + "...",
                "chapters": {},
                "passages": [],
                "references": []
            }

    def generate_content(self, prompt: str, url: str = None) -> str:
        """Wrapper method to maintain compatibility with previous interface"""
        response = self.prompt_model(messages=[{"role": "user", "content": prompt}])
        if isinstance(response, dict):
            return response.get("content", "")
        return response