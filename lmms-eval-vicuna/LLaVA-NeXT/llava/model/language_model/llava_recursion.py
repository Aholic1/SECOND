from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss

from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaModel, LlavaLlamaForCausalLM

from lmms_eval.recursion_utils import *
from lmms_eval.second_audit import (
    applied_selection_mask,
    binary_logit_record,
    parse_audit_condition,
    resolve_binary_token_ids,
    select_final_logits,
    uses_contrastive_decoding,
)
import numpy as np
import cv2
import torch.nn.functional as F
import csv
import os
import ast
from pathlib import Path
import math
import copy
import wandb
import json

from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    get_heatmap
)
from llava.mm_utils import (
    norm_relu,
    norm_min_max,
)

@dataclass
class LlavaRecursionConfig():    
    stages: List[int] = (-2, -1, 0, 1)
    positional_embedding_type: str = "bilinear"
    generation_type: str = "recursion"
    attention_thresholding_type: str = "layer_mean_topk"
    attention_threshold: List[float] = (0.9, 0.9, 0.9)
    contrastive_alphas: List[float] = (1.0, 1.0, 1.0)
    fix_grid: Optional[str] = "2x2"
    _device: str = "cuda:0"
    use_noised_for_contrastive: bool = False
    cd_strategy: str = "default"

class LlavaLlamaForRecursion(LlavaLlamaForCausalLM):
    config_class = LlavaRecursionConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
    
    def init_all(self, recursion_config, *args, **kwargs):
        # Initialize recursive generation attributes        
        self.stages = recursion_config.stages
        self.positional_embedding_type = recursion_config.positional_embedding_type
        self.generation_type = recursion_config.generation_type
        self.attention_thresholding_type = recursion_config.attention_thresholding_type
        self.attention_threshold = recursion_config.attention_threshold
        self.cd_strategy = recursion_config.cd_strategy

        self.contrastive_decoder = ContrastiveDecoder(self.cd_strategy)

        # Device assignment
        self._device = recursion_config._device

        # Image and patch size calculations
        self.patch_size = 14
        print(self.model)
        self.image_size = getattr(
            self.model.vision_tower.vision_tower.vision_model.embeddings, "image_size", None
        )
        if self.image_size is None:
            raise ValueError("`image_size` is not defined in vision tower embeddings.")

        self.smallest_grid_size = int(self.image_size * pow(2, self.stages[0]) // self.patch_size)
        self.largest_grid_size = int(self.image_size * pow(2, self.stages[-1]) // self.patch_size)

        # Initialize downsampled vision towers after loading weights
        if recursion_config.stages[0] < 0:
            self.model.downsampled_vision_towers = self.init_downsampled_vision_towers(
                self.model.vision_tower,
                recursion_config.stages,
                recursion_config.positional_embedding_type,
                self._device,
            )
        print("Downsampled towers initialized.")

        self.contrastive_alphas = recursion_config.contrastive_alphas    
        self.use_noised_for_contrastive = recursion_config.use_noised_for_contrastive
    
    def init_downsampled_vision_towers(self, vision_tower, stages, positional_embedding_type, device):
        print(f"change positional embedding to {positional_embedding_type}")
        downsampled_vision_towers = torch.nn.ModuleDict()
        for stage in stages:
            if stage == 0:
                break
            downsampled_vision_towers[str(stage)] = copy.deepcopy(vision_tower)

        # Default configurations of model position embedding
        patch_size = 14
        image_size = vision_tower.vision_tower.vision_model.embeddings.image_size

        for stage in stages:
            if stage == 0:
                break
            downsampled_image_size = image_size * pow(2, stage)
            assert (downsampled_image_size // patch_size) == (downsampled_image_size / patch_size), f"unavailable stage: {stage}"
            
            
            num_patches = int((downsampled_image_size // patch_size) ** 2)
            num_positions = num_patches + 1
            embed_dim = vision_tower.vision_tower.vision_model.embeddings.embed_dim

            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.image_size = downsampled_image_size
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.num_patches = num_patches
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.num_positions = num_positions
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)
        
            # Modify positional embedding to match the resized image size
            if positional_embedding_type == "zero":       
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                torch.nn.init.constant_(downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight, 0)
            elif positional_embedding_type == "interpolation":
                print("interpolate embedding type.")
                # Interpolate from the pretrained positional embedding
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                original_num_positions = original_embedding.size(0)
                new_embedding = torch.nn.functional.interpolate(
                    original_embedding.unsqueeze(0).transpose(1, 2), 
                    size=(num_positions,), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
            
            elif positional_embedding_type == "reduced":
                print("Reduced embedding type.")
                # Reduce the pretrained embedding by truncating
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(original_embedding[:num_positions])
                    
            elif positional_embedding_type == "bilinear_interpolation":
                # Interpolate from the pretrained positional embedding
                print("Bilienar interpolation embedding type.")
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                cls_token = original_embedding[:1, :]
                original_embedding = original_embedding[1:, :]  # Skip CLS
                # (1, 1024, 24, 24)
                new_height = int(len(original_embedding) ** (1/2) / (2 ** -stage))
                original_embedding = original_embedding.view(int(len(original_embedding)**(1/2)), int(len(original_embedding)**(1/2)), -1).permute(2, 0, 1).unsqueeze(0)
                
                resized_positional_embeddings = torch.nn.functional.interpolate(
                                                original_embedding,
                                                size=(new_height, new_height),  # r
                                                mode='bilinear',
                                                align_corners=False
                                            )  
                resized_positional_embeddings = resized_positional_embeddings.squeeze(0).permute(1, 2, 0).reshape(-1, 1024)
                new_embedding = torch.cat([cls_token, resized_positional_embeddings], dim=0)  
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(len(new_embedding), embed_dim).to(dtype=torch.float16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
                
        return downsampled_vision_towers
    
    def reset_image_mask(self):
        self.image_mask = dict()
        for idx, stage in enumerate(self.stages):
            if idx == 0:
                # activate every token of the first stage
                self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        
            else:
                self.image_mask[stage] = torch.zeros(self.largest_grid_size, self.largest_grid_size, device=self.device)
        self.pad_mask = dict()
    
    def set_pad_mask(self, size, origin_x, origin_y):
        self.pad_mask = dict()
        for stage in self.stages:
            grid = int(self.image_size * pow(2, stage) // self.patch_size)
            temp_tensor = torch.zeros(grid, grid, device=self.device)
            step = size / grid

            init_x, init_y = 0, 0
            stepped_x, stepped_y = 0, 0
            while stepped_x <= (size - origin_x) / 2:
                stepped_x += step
                init_x += 1
            while stepped_y <= (size - origin_y) / 2:
                stepped_y += step
                init_y += 1
            end_x, end_y = init_x, init_y
            while stepped_x < (size + origin_x) / 2:
                stepped_x += step
                end_x += 1
            while stepped_y < (size + origin_y) / 2:
                stepped_y += step
                end_y += 1

            temp_tensor[init_y - 1:end_y, init_x - 1:end_x] = 1
            temp_tensor = F.interpolate(temp_tensor.unsqueeze(0).unsqueeze(0), size=(self.largest_grid_size, self.largest_grid_size), mode='nearest').squeeze()
            self.pad_mask[stage] = temp_tensor


    def combine_image_and_pad_mask(self):
        for stage in self.stages:
            self.image_mask[stage] = self.image_mask[stage] * self.pad_mask[stage]
    
    def activate_every_image_masks(self):
        self.image_mask = dict()
        for stage in self.stages:
            self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        
    
    def activate_image_mask(self, stage):
        self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)

    @torch.no_grad()
    def generate_recursive(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,            
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]: 
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        attention_mask = kwargs.pop("attention_mask", None)   
        pad_token_ids = kwargs.pop("pad_token_ids", None)      
        downsampled_images = kwargs.pop("downsampled_images", None)
        max_length = kwargs.pop("max_length", None)
        gen_kwargs = kwargs.pop("gen_kwargs", None)
        tokenizer = kwargs.pop("tokenizer", None)
        question_input = kwargs.pop("question_input", None)
        audit_condition = parse_audit_condition(kwargs.pop("audit_condition", None))
        audit_recorder = kwargs.pop("audit_recorder", None)
        audit_sample_id = kwargs.pop("audit_sample_id", None)

        if audit_condition is not None and audit_recorder is None:
            raise ValueError("audit_recorder is required when audit_condition is set")
        binary_token_ids = None
        if audit_recorder is not None:
            binary_token_ids = resolve_binary_token_ids(tokenizer, question_input[0])

        if self.use_noised_for_contrastive:
            noised_images = add_diffusion_noise(images, 500)

        self.reset_image_mask() 
        final_token = []
        deferred_audit = None

        for token_idx in range(max_length):
            self.reset_image_mask()
            stage_logit_list = []
            audit_transitions = []
            audited_token = audit_recorder is not None and token_idx == audit_recorder.token_idx
            audit_provenance_traces = []

            for idx_stage, stage in enumerate(self.stages):
                # Necessary for unpadding
                # self.combine_image_and_pad_mask()

                last_stage = (stage == self.stages[-1])
                audit_mask_trace = {} if audited_token else None
                audit_provenance_trace = {} if audited_token else None

                cont = self.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_ids,
                    images=images,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=1,                    
                    generation_type=self.generation_type,
                    return_dict_in_generate=True,
                    output_attentions=(not last_stage),
                    output_scores=True,
                    downsampled_images=downsampled_images,
                    image_mask=self.image_mask,
                    audit_mask_trace=audit_mask_trace,
                    audit_provenance_trace=audit_provenance_trace,
                    audit_forward_stage=stage,
                )

                if audited_token:
                    audit_provenance_traces.append(audit_provenance_trace)

                # Delete the start-of-sequence (SOS) token if it exists
                if cont["sequences"][0][0] == 1:  # Assuming 1 is the SOS token ID
                    cont["sequences"] = cont["sequences"][0][1:].unsqueeze(0)

                text_outputs = tokenizer.batch_decode(cont["sequences"], skip_special_tokens=True)
                scores = cont.scores

                sequences = cont["sequences"][0]
                stage_logit_list.append(scores[0])
                if audited_token and audit_transitions:
                    previous_transition = audit_transitions[-1]
                    if previous_transition["target_stage"] == stage:
                        consumed_mask = audit_mask_trace.get(int(stage))
                        if consumed_mask is None:
                            raise ValueError(
                                f"model did not report a consumed mask for target stage {stage}"
                            )
                        previous_transition["consumed_mask"] = consumed_mask

                if last_stage:
                    # Contrastive decoding
                    final_logit = scores[0]

                    # use noised version of image for decoding
                    if self.use_noised_for_contrastive:
                        noised_cont = self.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            pad_token_id=pad_token_ids,
                            images=noised_images,
                            image_sizes=gen_kwargs["image_sizes"],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=1,                    
                            generation_type=None,
                            return_dict_in_generate=True,
                            output_attentions=False,
                            output_scores=True                            
                        )
                        noised_scores = noised_cont.scores[0]
                        final_logit += self.contrastive_alphas[-1]*(final_logit - noised_scores)
                    else:
                        use_cd = (
                            True
                            if audit_condition is None
                            else uses_contrastive_decoding(audit_condition)
                        )
                        final_logit = select_final_logits(
                            stage_logit_list,
                            self.contrastive_alphas,
                            use_cd=use_cd,
                        )

                    best_token = torch.argmax(final_logit, dim=-1)
                    final_token.append(best_token.item())

                    if audited_token:
                        deferred_audit = {
                            "stage_logits": list(stage_logit_list),
                            "final_logit": final_logit,
                            "transitions": audit_transitions,
                            "provenance_traces": audit_provenance_traces,
                        }
                    #stage_text = tokenizer.batch_decode(best_token, skip_special_tokens=True)[0]

                    # Append decoded text to final output
                    final_text = tokenizer.batch_decode([final_token], skip_special_tokens=True)[0]

                    # Update input_ids and attention_mask for the next token
                    input_ids = torch.cat([input_ids, best_token.unsqueeze(0)], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=-1
                    )
                    break

                ret_attn = get_heatmap(
                    self,
                    cont,
                    tokenizer,
                    question_input[0],
                    input_ids,
                    stage,
                    self.stages,
                    self.image_mask,
                    select_token=None,
                )

                if self.attention_thresholding_type == "layer_mean":
                    self.image_mask[stage + 1] = layer_mean_based_recursion(
                        attn=ret_attn,
                        attn_threshold=self.attention_threshold[idx_stage],
                        image_mask=self.image_mask[stage + 1]
                    )
                elif self.attention_thresholding_type == "layer_mean_topk":
                    self.image_mask[stage + 1] = layer_mean_topk_based_recursion(
                        attn=ret_attn,
                        top_k=self.attention_threshold[idx_stage],
                        image_mask=self.image_mask[stage + 1]
                    )
                elif self.attention_thresholding_type == "confidence_topk":
                    self.image_mask[stage + 1] = confidence_topk_based_recursion(
                        attn=ret_attn,
                        top_k=self.attention_threshold[idx_stage],
                        sequences=sequences,
                        scores=scores,
                        image_mask=self.image_mask[stage + 1]
                    )
                elif self.attention_thresholding_type == "attn_topk":
                    target_stage = self.stages[idx_stage + 1]
                    proposed_mask, calculated_threshold = attn_entropy_topk_based_recursion(
                        attn=ret_attn,
                        base_top_k=self.attention_threshold[idx_stage],
                        image_mask=self.image_mask[target_stage],
                    )
                    if audit_condition is None:
                        self.image_mask[target_stage] = proposed_mask
                    else:
                        applied_mask = applied_selection_mask(audit_condition, proposed_mask)
                        # STEP6_MASK_INTERVENTION
                        from lmms_eval.second_intervention import intervene_mask
                        applied_mask = intervene_mask(applied_mask, audit_sample_id, token_idx, target_stage, audit_condition)
                        self.image_mask[target_stage] = applied_mask

                        if audited_token:
                            audit_transitions.append({
                                "token_idx": token_idx,
                                "source_stage_index": idx_stage,
                                "source_stage": stage,
                                "target_stage": target_stage,
                                "ret_attn": ret_attn.detach(),
                                "proposed_mask": proposed_mask.detach(),
                                "applied_mask": applied_mask.detach(),
                                "calculated_threshold": calculated_threshold,
                            })
                else:
                    self.activate_image_mask(self.stages[idx_stage + 1])

                del cont
                torch.cuda.empty_cache()

            # Terminate if end-of-sequence (EOS) token is generated or max tokens reached
            if input_ids[0, -1] == tokenizer.eos_token_id:
                break

        if deferred_audit is not None:
            stage_margins = [
                binary_logit_record(logits, binary_token_ids)
                for logits in deferred_audit["stage_logits"]
            ]
            final_margin = binary_logit_record(
                deferred_audit["final_logit"], binary_token_ids
            )
            for transition in deferred_audit["transitions"]:
                source_index = transition.pop("source_stage_index")
                mask_before = torch.zeros_like(transition["proposed_mask"])
                (
                    recomputed_mask,
                    recomputed_threshold,
                    diagnostics,
                ) = attn_entropy_topk_based_recursion(
                    attn=transition["ret_attn"],
                    base_top_k=self.attention_threshold[source_index],
                    image_mask=mask_before.clone(),
                    return_diagnostics=True,
                )
                if not torch.equal(recomputed_mask, transition["proposed_mask"]):
                    raise ValueError(
                        "deferred audit diagnostics changed the proposed mask"
                    )
                if recomputed_threshold != transition["calculated_threshold"]:
                    raise ValueError(
                        "deferred audit diagnostics changed the selection threshold"
                    )
                applied_mask = transition["applied_mask"]
                applied_count = int(applied_mask.sum().item())
                total_count = int(applied_mask.numel())
                transition["mask_before"] = mask_before
                transition["metadata"] = {
                    **diagnostics,
                    "source_stage_margin": stage_margins[source_index],
                    "target_stage_margin": stage_margins[source_index + 1],
                    "proposed_selected_token_count": diagnostics[
                        "selected_token_count"
                    ],
                    "proposed_keep_ratio": diagnostics["actual_keep_ratio"],
                    "applied_selected_token_count": applied_count,
                    "applied_keep_ratio": applied_count / total_count,
                    "final_decoding_margin": final_margin,
                }
                transition.pop("calculated_threshold")
                audit_recorder.write_transition(
                    audit_sample_id,
                    **transition,
                )
            audit_recorder.write_provenance(
                audit_sample_id,
                token_idx=audit_recorder.token_idx,
                traces=deferred_audit["provenance_traces"],
            )

        if audit_recorder is not None:
            audit_recorder.finish_sample(
                audit_sample_id,
                {
                    "final_token_ids": final_token,
                    "final_decoded_response": final_text,
                },
            )

        return [final_text]
