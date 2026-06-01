import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from model import *
from utils.constants import DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from huggingface_hub import hf_hub_download




def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", cache_dir="../../checkpoint"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'grasp' in model_name.lower():
        # Load GraspLLM model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading GraspLLM from base model...')
            model = GraspLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional GraspLLM weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu', weights_only=False)
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu', weights_only=False)
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            print('Loading GraspLLM from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            _mb = model_base.lower()
            if ('qwen3_moe' in _mb) or ('qwen3-30b' in _mb) or ('a3b' in _mb):
                _Cls = GraspQwen3MoEForCausalLM
            elif 'qwen3' in _mb:
                _Cls = GraspQwen3ForCausalLM
            elif 'mistral' in _mb:
                _Cls = GraspMistralForCausalLM
            else:
                _Cls = GraspLlamaForCausalLM
            print(f"[builder] base_model={model_base}  -> {_Cls.__name__}")
            model, _loading_info = _Cls.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                cache_dir=cache_dir, output_loading_info=True, **kwargs)
            _PROJECTOR_NS = ("mm_projector", "special_token_emb")
            _is_proj = lambda k: any(ns in k for ns in _PROJECTOR_NS)
            _missing  = [k for k in (_loading_info.get("missing_keys")    or []) if not _is_proj(k)]
            _unexpect = [k for k in (_loading_info.get("unexpected_keys") or []) if not _is_proj(k)]
            if _missing or _unexpect:
                head = lambda lst: lst[:5] + (["..."] if len(lst) > 5 else [])
                raise RuntimeError(
                    f"[builder] backbone load report shows uncovered keys.\n"
                    f"  missing    ({len(_missing)})  : {head(_missing)}\n"
                    f"  unexpected ({len(_unexpect)}) : {head(_unexpect)}\n"
                    f"This typically means transformers' checkpoint-conversion-mapping for "
                    f"`config.model_type={cfg_pretrained.model_type!r}` is not registered for the "
                    f"Grasp* subclass.  See model/language_model/grasp_qwen3_moe.py for an "
                    f"example fix using register_checkpoint_conversion_mapping()."
                )
            print(f"[builder] backbone load OK  (no missing/unexpected outside mm_projector)")
            # -------------------------------------------------------------
            # model.get_model().initialize_graph_modules(cfg_pretrained)
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu', weights_only=False)
                print("Load from local path")
            else:
                from huggingface_hub import hf_hub_download
                model_path_hf = hf_hub_download(repo_id=model_path,  filename='mm_projector.bin')
                mm_projector_weights = torch.load(model_path_hf, map_location='cpu', weights_only=False)
                print("Load from huggingface")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}

            meta_path = os.path.join(model_path, 'mm_projector_meta.json')
            if os.path.isfile(meta_path):
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                wanted_cls = meta.get("backbone_class")
                got_cls = _Cls.__name__
                if wanted_cls and wanted_cls != got_cls:
                    raise RuntimeError(
                        f"[builder] mm_projector_meta.json says it was trained "
                        f"against backbone_class={wanted_cls!r}, but you are loading "
                        f"it into {got_cls!r}.  This usually means the wrong "
                        f"`--model_base` was passed.  Refusing to load silently.")
                wanted_hs = int(meta.get("hidden_size", -1))
                got_hs = int(getattr(model.config, "hidden_size", -1))
                if wanted_hs > 0 and got_hs > 0 and wanted_hs != got_hs:
                    raise RuntimeError(
                        f"[builder] mm_projector hidden_size mismatch: "
                        f"trained against hidden_size={wanted_hs}, loading into "
                        f"backbone with hidden_size={got_hs}.")
                # Shape-cross-check projector keys.
                got_keys = {k: tuple(v.shape) for k, v in mm_projector_weights.items()}
                want_keys = {k: tuple(v) for k, v in meta.get("key_shapes", {}).items()}
                for k, s in want_keys.items():
                    if k in got_keys and got_keys[k] != s:
                        raise RuntimeError(
                            f"[builder] mm_projector key {k!r} shape mismatch: "
                            f"meta={s} bin={got_keys[k]}")
                print(f"[builder] mm_projector_meta OK  step={meta.get('global_step')}  "
                      f"epoch={meta.get('epoch')}  keys={len(want_keys)}")

            missing_unexpected = model.load_state_dict(mm_projector_weights, strict=False)
            unexpected = list(getattr(missing_unexpected, "unexpected_keys", []) or [])
            if unexpected:
                print(f"[builder] WARNING: unexpected keys in mm_projector.bin: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = GraspLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, cache_dir=cache_dir, **kwargs)


    if 'grasp' in model_name.lower():
        mm_use_graph_start_end = getattr(model.config, "mm_use_graph_start_end", False)
        if mm_use_graph_start_end:
            tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
