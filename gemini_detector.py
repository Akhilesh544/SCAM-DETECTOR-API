import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(".env.local")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("===== AVAILABLE MODELS =====")
for m in client.models.list():
    print(m.name)
print("============================")


class GeminiDetector:

    def analyze_image(self, image_path: str):

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
You are an AI assistant that performs a preliminary visual inspection of Indian currency.

Important:
- You CANNOT determine authenticity with certainty from a single image.
- You should only evaluate visible features.
- If image quality or angle prevents verification, mention that.
- Never state that a note is definitely genuine or definitely counterfeit.

Verdict must be one of:
- Likely Genuine
- Suspicious
- Unable to Determine

Return ONLY valid JSON.

{
  "denomination":"",
  "verdict":"",
  "confidence":0,
  "summary":"",
  "reasons":[]
}
"""

        response = client.models.generate_content(
           model="models/gemini-flash-latest",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ]
        )
        for model in client.models.list():
         print(model.name)


        text = response.text.strip()

        if text.startswith("```"):
            text = (
                text.replace("```json", "")
                    .replace("```", "")
                    .strip()
            )

        return json.loads(text)