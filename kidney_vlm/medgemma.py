from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass
class MedGemmaConfig:
    model_id: str = "google/medgemma-1.5-4b-it"
    dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"


class MedGemma:
    def __init__(self, config: MedGemmaConfig):
        self.config = config
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=config.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(config.model_id)

    def _move_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors to model device without breaking integer tensors.

        Important: input_ids must stay integer dtype. Only float tensors (pixel_values)
        should be cast to bf16/fp16.
        """
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    out[k] = v.to(self.model.device, dtype=self.config.dtype)
                else:
                    out[k] = v.to(self.model.device)
            else:
                out[k] = v
        return out

    @torch.inference_mode()
    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text conditioned on 1+ images."""
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = self._move_inputs(dict(inputs))

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

        input_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, **gen_kwargs)
        gen_tokens = out[0][input_len:]
        text = self.processor.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()
