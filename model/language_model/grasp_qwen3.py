from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen3Config, Qwen3Model, Qwen3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..grasp_arch import GraspMetaModel, GraspMetaForCausalLM
from utils.constants import IGNORE_INDEX


class GraspQwen3Config(Qwen3Config):
    model_type = "grasp_qwen3"


class GraspQwen3Model(GraspMetaModel, Qwen3Model):
    config_class = GraspQwen3Config

    def __init__(self, config: Qwen3Config):
        super(GraspQwen3Model, self).__init__(config)


class GraspQwen3ForCausalLM(Qwen3ForCausalLM, GraspMetaForCausalLM):
    config_class = GraspQwen3Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = GraspQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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


AutoConfig.register("grasp_qwen3", GraspQwen3Config)
AutoModelForCausalLM.register(GraspQwen3Config, GraspQwen3ForCausalLM)
