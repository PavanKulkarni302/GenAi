"""
ai_local.py
Hugging Face pipeline wrapper for summarization and generation.
This wrapper attempts to initialize HF pipelines lazily. If model loading fails
(e.g., no model downloaded), a simple fallback strategy is used.
"""
from transformers import pipeline
import logging

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # CPU-friendly summarization/generation model

class LocalAI:
    def __init__(self, model_name=MODEL_NAME, device=-1):
        self.model_name = model_name
        self.device = device
        self._summarizer = None
        self._generator = None

    def _init_summarizer(self):
        if self._summarizer is None:
            try:
                self._summarizer = pipeline("summarization", model=self.model_name, device=self.device)
            except Exception as e:
                LOGGER.warning("Failed to load summarization pipeline: %s", e)
                self._summarizer = None

    def _init_generator(self):
        if self._generator is None:
            try:
                self._generator = pipeline("text2text-generation", model=self.model_name, device=self.device)
            except Exception as e:
                LOGGER.warning("Failed to load generation pipeline: %s", e)
                self._generator = None

    def summarize(self, text, max_length=120, min_length=30):
        """Summarize text using HF pipeline or fallback rule-based summary."""
        self._init_summarizer()
        if self._summarizer:
            try:
                out = self._summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
                return out[0]["summary_text"].strip()
            except Exception as e:
                LOGGER.warning("Summarizer runtime error: %s", e)

        # Fallback: return first 2 sentences or first 200 chars
        import re
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sents) >= 2:
            return ' '.join(sents[:2])
        return text.strip()[:200]

    def generate(self, prompt, max_length=150):
        """Generate text (e.g., onboarding steps or email draft)."""
        self._init_generator()
        if self._generator:
            try:
                out = self._generator(prompt, max_length=max_length, do_sample=False)
                # generated_text for text2text-generation, sometimes 'summary_text' for summarizers
                key = "generated_text" if "generated_text" in out[0] else list(out[0].values())[0]
                return out[0].get("generated_text", key).strip()
            except Exception as e:
                LOGGER.warning("Generator runtime error: %s", e)

        # Fallback generation: simple template expansion
        return prompt + "\n\n[Note: model not available — this is a fallback placeholder.]"
