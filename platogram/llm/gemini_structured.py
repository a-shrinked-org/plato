# In platogram/llm/gemini_structured.py

from platogram.llm.gemini import Gemini
import json

class StructuredGemini(Gemini):
    def generate_structured_output(self, transcript: str):
        prompt = (
            "Extract structured information from the following transcript. "
            "Return a JSON object with 'title', 'abstract', and 'passages'.\\n\\n"
            "Transcript:\\n" + transcript
        )
        raw_output = self.generate_content(prompt)
        try:
            output = json.loads(raw_output)
            return {
                "title": output.get("title", "Generated Document"),
                "abstract": output.get("abstract", "Transcript processed. (No structured output available)"),
                "passages": output.get("passages", [])
            }
        except json.JSONDecodeError:
            return {
                "title": "Generated Document",
                "abstract": "Transcript processed. (Structured output not available with Gemini.)",
                "passages": []
            }
