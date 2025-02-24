# In platogram/llm/gemini_structured.py

import re
import json
from platogram.llm.gemini import Model

class StructuredGemini(Model):
    def generate_structured_output(self, transcript: str):
        # System prompt with clear role and task definitions, similar to Sonnet setup
        system_prompt = """
        <role>
        You are a skilled academic editor analyzing content.
        Your task is to extract a clear title, comprehensive abstract, and key passages from the transcript.
        </role>
        <task>
        1. Study the provided transcript carefully.
        2. Generate a concise title that captures the essence of the content.
        3. Create a comprehensive abstract that summarizes the main points.
        4. Identify and list the key passages that represent the core ideas.
        5. Return the results in a JSON object with 'title', 'abstract', and 'passages' fields.
        </task>
        """

        # Few-shot examples to guide the model
        few_shot_examples = """
        Example 1:
        Transcript: "This is a sample transcript about quantum computing. It explains how quantum computers use quantum bits to perform calculations faster..."
        Output: {
            "title": "Introduction to Quantum Computing",
            "abstract": "This transcript provides an overview of quantum computing, including its principles and applications.",
            "passages": ["Quantum computing leverages quantum bits...", "It has the potential to revolutionize..."]
        }

        Example 2:
        Transcript: "Another transcript discussing artificial intelligence and its recent advancements in natural language processing..."
        Output: {
            "title": "Advancements in Artificial Intelligence",
            "abstract": "This transcript explores recent developments in AI and their impact on various industries.",
            "passages": ["AI has made significant strides in natural language processing...", "Ethical considerations are becoming increasingly important..."]
        }
        """

        # User prompt combining few-shot examples and the transcript
        user_prompt = f"""
        {few_shot_examples}

        Now, analyze the following transcript and generate the structured output as shown in the examples:

        Transcript:
        {transcript}
        """

        # Call the model with the enhanced prompt
        raw_output = self.prompt_model(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        try:
            if isinstance(raw_output, dict):
                return {
                    "title": raw_output.get("title", "Generated Document"),
                    "abstract": raw_output.get("abstract", "Transcript processed. (No structured output available)"),
                    "passages": raw_output.get("passages", [])
                }
            # Strip backticks and parse JSON if returned as a string
            if isinstance(raw_output, str):
                raw_output = re.sub(r'^```json\s*|\s*```$', '', raw_output, flags=re.MULTILINE).strip()
                output = json.loads(raw_output)
                return {
                    "title": output.get("title", "Generated Document"),
                    "abstract": output.get("abstract", "Transcript processed. (No structured output available)"),
                    "passages": output.get("passages", [])
                }
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error processing Gemini output: {e}")
            return {
                "title": "Generated Document",
                "abstract": "Transcript processed. (Structured output not available with Gemini.)",
                "passages": []
            }

    def generate_content(self, prompt: str, url: str = None) -> str:
        """Wrapper method to maintain compatibility with previous interface"""
        response = self.prompt_model(messages=[{"role": "user", "content": prompt}])
        if isinstance(response, dict):
            return response.get("content", "")
        return response