# Qwen3.5-9B-Instruct-Turca-TurkishLLM 🇹🇷

A supervised fine-tune of **Qwen/Qwen3.5-9B** for Turkish instruction following, reasoning, and natural language generation — trained on 500K samples from the InstrucTurca dataset.

> 📦 **Model weights, training details, and full documentation are on Hugging Face:**
> **[👉 muhammedksee/Qwen3.5-9B-Instruct-Turca-TurkishLLM](https://huggingface.co/muhammedksee/Qwen3.5-9B-Instruct-Turca-TurkishLLM)**
>
> 🗜️ **Local inference (GGUF / Ollama / llama-cpp):**
> **[👉 muhammedksee/Qwen3.5-9B-Instruct-Turca-TurkishLLM-GGUF](https://huggingface.co/muhammedksee/Qwen3.5-9B-Instruct-Turca-TurkishLLM-GGUF)**

---

## What is this?

| Field | Value |
| --- | --- |
| **Base Model** | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| **Method** | SFT + LoRA |
| **Dataset** | InstrucTurca — 500K samples |
| **Languages** | Turkish (primary), English |
| **Hardware** | NVIDIA H100 80GB |
| **License** | Apache 2.0 |

---

## Benchmark Results

Evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.2 — 0-shot, H100 GPU.

**English (Catastrophic Forgetting Test)**

| Task | Metric | Score |
| --- | --- | --- |
| MMLU | acc | **0.7787** |
| HellaSwag | acc_norm | **0.7834** |
| ARC-Challenge | acc_norm | **0.5375** |

**Turkish NLP**

| Task | Metric | Score |
| --- | --- | --- |
| Belebele (TR) | acc | **0.8144** |
| Turkish MMLU | acc | **0.6555** |
| XCOPA (TR) | acc | **0.6780** |

No catastrophic forgetting observed — English reasoning is preserved after Turkish fine-tuning.

---

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "muhammedksee/Qwen3.5-9B-Instruct-Turca-TurkishLLM"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "Sen yardımcı, dürüst ve zararsız bir Türkçe yapay zeka asistanısın."},
    {"role": "user",   "content": "Türkiye'nin en büyük şehri hangisidir?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
outputs = [o[inputs.input_ids.shape[1]:] for o in outputs]
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

---

## Contact

- 🔗 [LinkedIn](https://linkedin.com/in/muhammedksee)
- 🐙 [GitHub](https://github.com/muhammedksee)
- 🤗 [Hugging Face](https://huggingface.co/muhammedksee)
