import os
import json
import boto3
from typing import Optional, Dict, Any
from flags import FLAGS


class BedrockScamDetector:
    """
    BedrockScamDetector uses Amazon Bedrock for text and image analysis.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-v2:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        aws_bearer_token_bedrock: Optional[str] = None,
    ):
        self.model_id = model_id

        if aws_bearer_token_bedrock:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_bearer_token_bedrock
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
            )

        elif aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

        else:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
            )

    def _extract_json(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        return text

    def analyze(self, text: str) -> Dict[str, Any]:

        text = text.strip()

        if not text:
            return {
                "verdict": "UNKNOWN",
                "trust_score": None,
                "scam_probability": None,
                "reasons": [],
            }

        flags_desc = "\n".join(
            [
                f"- '{flag.name}': {flag.explanation} (Clues: {flag.pattern})"
                for flag in FLAGS
            ]
        )

        system_prompt = f"""
You are an expert security system.

Analyze the following message.

Possible scam indicators:

{flags_desc}

Return ONLY JSON.

{{
  "verdict":"",
  "trust_score":0,
  "scam_probability":0,
  "reasons":[]
}}
"""

        try:

            response = self.client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": text
                            }
                        ]
                    }
                ],
                system=[
                    {
                        "text": system_prompt
                    }
                ]
            )

            raw = response["output"]["message"]["content"][0]["text"]

            clean = self._extract_json(raw)

            result = json.loads(clean)

            return result

        except Exception as e:

            return {
                "verdict": "UNKNOWN",
                "trust_score": None,
                "scam_probability": None,
                "reasons": [
                    {
                        "flag": "error",
                        "matched": "",
                        "why": str(e),
                    }
                ],
            }

    def analyze_image(self, image_bytes: bytes, image_format: str = "jpeg"):

        prompt = """
You are an expert in Indian currency verification.

Analyze the uploaded Indian banknote.

Inspect:

- denomination
- watermark
- security thread
- microprinting
- color consistency
- print quality
- visible tampering

Do NOT claim absolute certainty.

Return ONLY valid JSON.

{
  "denomination":"",
  "verdict":"",
  "confidence":0,
  "summary":"",
  "reasons":[]
}
"""

        response = self.client.converse(
            modelId="anthropic.claude-3-5-sonnet-v2:0",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image_format,
                                "source": {
                                    "bytes": image_bytes
                                }
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        output = response["output"]["message"]["content"][0]["text"]

        clean = self._extract_json(output)

        return json.loads(clean)