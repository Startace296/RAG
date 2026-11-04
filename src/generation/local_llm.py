import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from ..core import env_flag, quiet_output


DEFAULT_LOCAL_LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
transformers_logging.set_verbosity_error()


class LocalLLM:
    """Lazy-loaded local causal LLM (Qwen by default) used to generate answers."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_LLM_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.enable_thinking = env_flag("QWEN_ENABLE_THINKING")
        self.load_in_4bit = env_flag("LLM_LOAD_IN_4BIT") and self.device == "cuda"

    def load(self) -> None:
        """Load tokenizer and model weights once."""
        if self.tokenizer is not None and self.model is not None:
            return

        with quiet_output():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                    device_map={"": 0},
                )
            else:
                # bfloat16 on CPU halves RAM; a 4B model in float32 needs ~16GB.
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.bfloat16,
                ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a completion for a user prompt, optionally with a system prompt."""
        self.load()
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Local LLM was not loaded.")

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        do_sample = env_flag("QWEN_DO_SAMPLE")
        generation_kwargs = {
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
            "do_sample": do_sample,
            "repetition_penalty": float(os.getenv("QWEN_REPETITION_PENALTY", "1.0")),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # Prefill long prompts in chunks: full-length attention on a 4GB GPU spills
        # into shared memory and makes every step many times slower.
        prefill_chunk_size = int(os.getenv("LLM_PREFILL_CHUNK_SIZE", "512"))
        if prefill_chunk_size > 0 and self.device == "cuda":
            generation_kwargs["prefill_chunk_size"] = prefill_chunk_size
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.3")),
                    "top_p": float(os.getenv("QWEN_TOP_P", "0.8")),
                    "top_k": int(os.getenv("QWEN_TOP_K", "20")),
                }
            )

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
