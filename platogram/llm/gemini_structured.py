# In platogram/llm/gemini_structured.py

from platogram.llm.gemini import Model
import json

class StructuredGemini(Model):
    def generate_structured_output(self, transcript: str):
        prompt = (
            "Extract structured information from the following transcript. "
            "Return a JSON object with 'title', 'abstract', and 'passages'.\n\n"
            "Transcript:\n" + transcript
        )
        raw_output = self.prompt_model(messages=[{"role": "user", "content": prompt}])
        try:
            if isinstance(raw_output, dict):
                return {
                    "title": raw_output.get("title", "Generated Document"),
                    "abstract": raw_output.get("abstract", "Transcript processed. (No structured output available)"),
                    "passages": raw_output.get("passages", [])
                }
            output = json.loads(raw_output)
            return {
                "title": output.get("title", "Generated Document"),
                "abstract": output.get("abstract", "Transcript processed. (No structured output available)"),
                "passages": output.get("passages", [])
            }
        except (json.JSONDecodeError, AttributeError):
            return {
                "title": "Generated Document",
                "abstract": "Transcript processed. (Structured output not available with Gemini.)",
                "passages": []
            }

    def generate_content(self, prompt: str, url: str = None) -> str:
        """Wrapper method to maintain compatibility with previous interface"""
        response = self.prompt_model(messages=[{"role": "user", "content": prompt}])
        if isinstance(response, dict):
            # Handle function call responses
            return response.get("content", "")
        return response