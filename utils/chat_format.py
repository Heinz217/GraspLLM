from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple

from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN

GRAPH_PLACEHOLDER_CHAR = "\uE000"

_VICUNA_V1_TEMPLATE = r"""{%- set system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions." -%}
{{- system_prompt + " " -}}
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{- 'USER: ' + message['content'] -}}
{%- elif message['role'] == 'assistant' -%}
{{- ' ASSISTANT:' -}}{%- generation -%}{{- message['content'] + eos_token -}}{%- endgeneration -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- ' ASSISTANT:' -}}
{%- endif -%}
"""

_MISTRAL_INST_TEMPLATE = r"""{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{- '[INST] ' + message['content'] + ' [/INST]' -}}
{%- elif message['role'] == 'assistant' -%}
{%- generation -%}{{- message['content'] + eos_token -}}{%- endgeneration -%}
{%- endif -%}
{%- endfor -%}
"""

_LLAMA3_INSTRUCT_TEMPLATE = r"""{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' -}}
{%- elif message['role'] == 'assistant' -%}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}{%- generation -%}{{- message['content'] + '<|eot_id|>' -}}{%- endgeneration -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
{%- endif -%}
"""

_QWEN3_CHATML_TEMPLATE = r"""{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' -}}
{%- elif message['role'] == 'assistant' -%}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}{%- generation -%}{{- message['content'] + '<|im_end|>' -}}{%- endgeneration -%}{{- '\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}
{%- endif -%}
"""


_BACKBONE_TO_TEMPLATE = {
    "vicuna":     _VICUNA_V1_TEMPLATE,
    "llama":      _VICUNA_V1_TEMPLATE,      # alias for "Llama wrapper class"
    "mistral":    _MISTRAL_INST_TEMPLATE,
    "llama3":     _LLAMA3_INSTRUCT_TEMPLATE,
    "qwen3":      _QWEN3_CHATML_TEMPLATE,
    "qwen3_moe":  _QWEN3_CHATML_TEMPLATE,
}


def _infer_backbone_kind(model_name_or_path: str) -> str:
    """Map a base-model path to one of our template keys."""
    p = model_name_or_path.lower()
    if ("qwen3_moe" in p) or ("qwen3-30b" in p) or ("a3b" in p):
        return "qwen3_moe"
    if "qwen3" in p:
        return "qwen3"
    if "mistral" in p:
        return "mistral"
    if "llama-3" in p or "llama3" in p:
        return "llama3"
    if "llama" in p or "vicuna" in p:
        return "vicuna"
    # Fallback: vicuna template is a safe-ish minimal chat format.
    return "vicuna"


def install_chat_template(tokenizer, backbone_kind: Optional[str] = None,
                          model_name_or_path: Optional[str] = None) -> str:
    """Install our jinja chat template onto the tokenizer (idempotent).

    Either provide `backbone_kind` directly, or `model_name_or_path` and
    we'll infer it.  Returns the kind we ended up using.
    """
    if backbone_kind is None:
        if model_name_or_path is None:
            raise ValueError("install_chat_template: provide backbone_kind or model_name_or_path")
        backbone_kind = _infer_backbone_kind(model_name_or_path)
    if backbone_kind not in _BACKBONE_TO_TEMPLATE:
        raise ValueError(f"install_chat_template: unsupported backbone_kind={backbone_kind!r}")
    tokenizer.chat_template = _BACKBONE_TO_TEMPLATE[backbone_kind]
    return backbone_kind


def _ids_from_chat_template(tokenizer, msgs, *, add_generation_prompt: bool,
                            return_mask: bool):
    """Run apply_chat_template and return (input_ids: list[int], mask: list[int] or None).

    Handles the multiple return shapes HF can produce:
      * plain list[int]
      * list[list[int]] (batched=False but still nested)
      * BatchEncoding with attribute access
      * dict with 'input_ids' / 'assistant_masks'
    """
    if return_mask:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
        mask = out["assistant_masks"] if isinstance(out, dict) else out.assistant_masks
    else:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )
        ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
        mask = None

    # Normalize: we want a flat list[int].  ids may be wrapped one or two
    # levels deep depending on transformers version.
    def _flatten(x):
        # tensor?
        if hasattr(x, "tolist"):
            x = x.tolist()
        # nested list -> first sample
        while isinstance(x, list) and x and isinstance(x[0], list):
            x = x[0]
        return list(x)

    ids = _flatten(ids)
    if mask is not None:
        mask = _flatten(mask)
        assert len(mask) == len(ids), f"mask len {len(mask)} != ids len {len(ids)}"
    return ids, mask


def _locate_placeholders_via_text(tokenizer, msgs, add_generation_prompt: bool
                                  ) -> Tuple[str, List[Tuple[int, int]]]:
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Find all placeholder char positions in text.
    positions = []
    start = 0
    while True:
        idx = text.find(GRAPH_PLACEHOLDER_CHAR, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1
    return text, positions


def build_supervised(tokenizer,
                     conversations: List[Dict[str, str]],
                     has_graph: bool = True,
                     model_max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    msgs = []
    for turn in conversations:
        role = role_map.get(turn["from"], turn["from"])
        content = turn["value"]
        if has_graph and role == "user":
            content = content.replace(DEFAULT_GRAPH_TOKEN, GRAPH_PLACEHOLDER_CHAR)
        msgs.append({"role": role, "content": content})

    ids, mask = _ids_from_chat_template(tokenizer, msgs,
                                         add_generation_prompt=False,
                                         return_mask=True)

    if has_graph:
        text, char_positions = _locate_placeholders_via_text(tokenizer, msgs,
                                                              add_generation_prompt=False)
        new_ids: List[int] = []
        new_mask: List[int] = []
        consumed_token = 0
        for char_idx in char_positions:
            target_text_len = len(text[:char_idx])
            lo, hi = consumed_token, len(ids)
            while lo < hi:
                mid = (lo + hi) // 2
                d = tokenizer.decode(ids[:mid], skip_special_tokens=False)
                if len(d) < target_text_len:
                    lo = mid + 1
                else:
                    hi = mid
            tok_start = lo
            lo2, hi2 = tok_start, len(ids)
            while lo2 < hi2:
                mid = (lo2 + hi2) // 2
                d = tokenizer.decode(ids[:mid], skip_special_tokens=False)
                if len(d) < target_text_len + 1:
                    lo2 = mid + 1
                else:
                    hi2 = mid
            tok_end = lo2

            new_ids.extend(ids[consumed_token:tok_start])
            new_mask.extend(mask[consumed_token:tok_start])
            new_ids.append(GRAPH_TOKEN_INDEX)
            new_mask.append(0)
            consumed_token = tok_end
        # Trailing tokens after the last placeholder.
        new_ids.extend(ids[consumed_token:])
        new_mask.extend(mask[consumed_token:])
        ids, mask = new_ids, new_mask

    labels = [(t if m_ else IGNORE_INDEX) for t, m_ in zip(ids, mask)]

    if model_max_length is not None and len(ids) > model_max_length:
        ids    = ids[:model_max_length]
        labels = labels[:model_max_length]

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels":    torch.tensor(labels, dtype=torch.long),
    }


def build_eval_prompt(tokenizer,
                      user_text: str,
                      has_graph: bool = True,
                      max_length: Optional[int] = None) -> torch.Tensor:
    """Return the prompt token ids ending right before the assistant turn."""
    content = user_text
    if has_graph:
        content = content.replace(DEFAULT_GRAPH_TOKEN, GRAPH_PLACEHOLDER_CHAR)
    msgs = [{"role": "user", "content": content}]
    ids, _ = _ids_from_chat_template(tokenizer, msgs,
                                      add_generation_prompt=True,
                                      return_mask=False)

    if has_graph:
        text, char_positions = _locate_placeholders_via_text(tokenizer, msgs,
                                                              add_generation_prompt=True)
        new_ids: List[int] = []
        consumed_token = 0
        for char_idx in char_positions:
            target_text_len = len(text[:char_idx])
            lo, hi = consumed_token, len(ids)
            while lo < hi:
                mid = (lo + hi) // 2
                d = tokenizer.decode(ids[:mid], skip_special_tokens=False)
                if len(d) < target_text_len:
                    lo = mid + 1
                else:
                    hi = mid
            tok_start = lo
            lo2, hi2 = tok_start, len(ids)
            while lo2 < hi2:
                mid = (lo2 + hi2) // 2
                d = tokenizer.decode(ids[:mid], skip_special_tokens=False)
                if len(d) < target_text_len + 1:
                    lo2 = mid + 1
                else:
                    hi2 = mid
            tok_end = lo2
            new_ids.extend(ids[consumed_token:tok_start])
            new_ids.append(GRAPH_TOKEN_INDEX)
            consumed_token = tok_end
        new_ids.extend(ids[consumed_token:])
        ids = new_ids

    if max_length is not None and len(ids) > max_length:
        ids = ids[-max_length:]

    return torch.tensor(ids, dtype=torch.long)


def stop_token_ids(tokenizer) -> List[int]:
    stops = set()
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))
    for s in ("<|eot_id|>", "<|im_end|>", "</s>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                stops.add(int(tid))
        except Exception:
            pass
    return sorted(stops)
