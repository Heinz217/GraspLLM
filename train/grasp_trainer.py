import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class GraspTrainer(Trainer):

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not has_length(dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler(train_dataset)

    def _dump_mm_projector(self, run_dir: str):
        if not (self.args.local_rank == 0 or self.args.local_rank == -1):
            return

        keys_to_match = ['mm_projector']
        if getattr(self.args, "use_graph_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        if getattr(self.args, "mm_use_graph_special_token", False):
            keys_to_match.extend(['special_token_emb'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            self.model.named_parameters(), keys_to_match
        )

        os.makedirs(run_dir, exist_ok=True)
        self.model.config.save_pretrained(run_dir)
        torch.save(weight_to_save, os.path.join(run_dir, 'mm_projector.bin'))

        import json
        cfg = self.model.config
        meta = {
            "schema_version": 1,
            "global_step": int(self.state.global_step),
            "epoch": float(self.state.epoch) if self.state.epoch is not None else None,
            "backbone_class": type(self.model).__name__,
            "backbone_path": getattr(cfg, "_name_or_path", None),
            "mm_hidden_size": int(getattr(cfg, "mm_hidden_size", -1)),
            "mm_projector_type": str(getattr(cfg, "mm_projector_type", "")),
            "hidden_size": int(getattr(cfg, "hidden_size", -1)),
            "vocab_size": int(getattr(cfg, "vocab_size", -1)),
            "key_count": len(weight_to_save),
            "key_shapes": {k: list(v.shape) for k, v in weight_to_save.items()},
        }
        with open(os.path.join(run_dir, 'mm_projector_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            ckpt_dir = os.path.join(run_dir, checkpoint_folder)

            self._dump_mm_projector(run_dir)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.config.save_pretrained(ckpt_dir)
        else:
            super(GraspTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            run_dir = output_dir if output_dir is not None else self.args.output_dir
            self._dump_mm_projector(run_dir)
        else:
            super(GraspTrainer, self)._save(output_dir, state_dict)
