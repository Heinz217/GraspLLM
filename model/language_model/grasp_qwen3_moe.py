from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (AutoConfig, AutoModelForCausalLM,
                          Qwen3MoeConfig, Qwen3MoeModel, Qwen3MoeForCausalLM)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    load_balancing_loss_func as _qwen3_moe_lb_loss,
)

from ..grasp_arch import GraspMetaModel, GraspMetaForCausalLM
from utils.constants import IGNORE_INDEX


class GraspQwen3MoEConfig(Qwen3MoeConfig):
    model_type = "grasp_qwen3_moe"


class GraspQwen3MoEModel(GraspMetaModel, Qwen3MoeModel):
    config_class = GraspQwen3MoEConfig

    def __init__(self, config: Qwen3MoeConfig):
        super(GraspQwen3MoEModel, self).__init__(config)


class GraspQwen3MoEForCausalLM(Qwen3MoeForCausalLM, GraspMetaForCausalLM):
    config_class = GraspQwen3MoEConfig

    def __init__(self, config):
        super(Qwen3MoeForCausalLM, self).__init__(config)
        self.model = GraspQwen3MoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # MoE-specific knobs (match HF defaults)
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.001)
        self.num_experts = getattr(config, "num_experts", None)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, graph, graph_emb)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Optional MoE load-balancing auxiliary loss.
        aux_loss = None
        if output_router_logits and getattr(outputs, "router_logits", None) is not None:
            aux_loss = _qwen3_moe_lb_loss(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if aux_loss is not None:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # See grasp_llama.py — handle DynamicCache truthiness on transformers 5.x.
        if past_key_values is not None:
            try:
                _has_kv = past_key_values.get_seq_length() > 0
            except AttributeError:
                _has_kv = bool(past_key_values)
            if _has_kv:
                input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "graph": kwargs.get("graph", None),
            "graph_emb": kwargs.get("graph_emb", None),
        })
        return model_inputs


AutoConfig.register("grasp_qwen3_moe", GraspQwen3MoEConfig)
AutoModelForCausalLM.register(GraspQwen3MoEConfig, GraspQwen3MoEForCausalLM)


# ---------------------------------------------------------------------------
# Checkpoint conversion mapping
#
# transformers 5.x ships its Qwen3-MoE implementation with a *fused* expert
# weight layout: each MLP layer holds two big 3D tensors,
#     experts.gate_up_proj  shape (num_experts, 2 * intermediate, hidden)
#     experts.down_proj     shape (num_experts, hidden, intermediate)
# whereas Qwen3-30B-A3B-Instruct's safetensors store experts un-fused, as
#     experts.{i}.gate_proj.weight
#     experts.{i}.up_proj.weight
#     experts.{i}.down_proj.weight
# A `register_checkpoint_conversion_mapping("qwen3_moe", ...)` is shipped
# in `transformers/conversion_mapping.py` to bridge the two layouts when
# loading any model whose `config.model_type == "qwen3_moe"`.
#
# Our subclass uses model_type = "grasp_qwen3_moe", so the conversion is
# never triggered, and ALL 128 experts in every one of the 48 layers end
# up *randomly initialized*.  Training silently fits mm_projector against
# random experts (loss looks fine because mm_projector overpowers
# everything else through 4096->2048 + 5 attention sinks), but at eval
# time the model spits pure garbage (alerts/中国梦/...) -- the random
# experts have no language modeling capacity.
#
# Fix: register the SAME conversion under our model_type at import time.
def _register_grasp_qwen3_moe_conversion_mapping() -> None:
    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        # Older transformers (<5.x) used a different mechanism that we
        # don't need to patch here -- they didn't fuse experts.
        return
    upstream = get_checkpoint_conversion_mapping("qwen3_moe")
    if upstream is None:
        # transformers updated and removed the fused layout; nothing to do.
        return
    try:
        register_checkpoint_conversion_mapping(
            "grasp_qwen3_moe", upstream, overwrite=True)
    except Exception as e:
        import warnings
        warnings.warn(f"[grasp_qwen3_moe] failed to register conversion mapping: {e}")


_register_grasp_qwen3_moe_conversion_mapping()
