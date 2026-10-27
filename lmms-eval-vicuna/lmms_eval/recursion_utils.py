import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F

import csv
from pathlib import Path

def calculate_entropy_for_attn_threshold(attn_map):
    flattened_attn = attn_map.view(-1)
    flattened_attn = flattened_attn / flattened_attn.sum()
    
    log_probs = torch.log(flattened_attn + 1e-8)
    
    # Calculate Entropy
    entropy = -torch.sum(flattened_attn * log_probs)
    return entropy.item()

def entropy_based_threshold(attn_map, base_threshold=0.2, scaling_factor=2, max_entropy=6.0):
    # Calculate Entropy
    entropy = calculate_entropy_for_attn_threshold(attn_map)
    normalized_entropy = min(entropy / max_entropy, 1.0)
    
    # Use a sensitive scaling function to make normalized_entropy more impactful
    sensitive_entropy = torch.exp(torch.tensor(normalized_entropy * scaling_factor) - 1)  # Shift to center around 1
    
    print(f'Calculated Sensitive Entropy: {sensitive_entropy}')  
    
    # Modify the threshold based on the sensitive entropy
    threshold = base_threshold * (1 - 0.5 * sensitive_entropy)  # Smaller factor for smaller changes
    threshold = min(max(threshold, 0.1), 0.9)  # Ensure it stays within range
    return threshold

def confidence_based_threshold(cumulative_confidences, base_threshold=0.2):
    # Take the last value from cumulative_confidences as the final confidence
    final_confidence = cumulative_confidences[-1]

    # Use exponential scaling for sensitivity
    sensitive_confidence = torch.exp(torch.tensor(final_confidence))  # Center around 1
    
    print(f'Calculated Sensitive Confidence: {sensitive_confidence}')  
    
    # Modify the threshold based on the sensitive confidence
    threshold = base_threshold * sensitive_confidence  # Increase threshold as confidence rises
    threshold = min(max(threshold, 0.1), 0.9)  # Ensure threshold stays within range
    return threshold

def calculate_entropy_and_all_confidences(sequence, scores):
    """
    Calculate entropy, full sequence confidence, and per-token cumulative confidences.
    Args:
        sequence (torch.Tensor): Generated sequence of token IDs.
        scores (list of torch.Tensor): List of logit tensors, one for each token in the sequence.

    Returns:
        tuple: Full sequence confidence, entropy, and a list of per-token cumulative confidences.
    """
    log_prob_sum_full = 0.0  # Log-probability sum for calculating full sequence confidence
    entropy_sum = 0.0  # Sum of entropies
    cumulative_confidences = []  # List to store cumulative confidence up to each token

    for idx, token_id in enumerate(sequence):
        probs = F.softmax(scores[idx], dim=-1)  # Softmax to get probabilities for the current token
        token_prob = probs[0, token_id].item()  # Probability of the actual token

        # Update cumulative log probability for the full sequence up to this token
        log_prob_sum_full += np.log(token_prob + 1e-10)
        # Calculate and store cumulative confidence for this subsequence
        cumulative_confidences.append(np.exp(log_prob_sum_full))
        
        # Entropy calculation for the token
        entropy_sum -= token_prob * np.log(token_prob + 1e-10)

    # Full sequence confidence (cumulative up to the last token)
    P_T_given_I_Q_full = cumulative_confidences[-1] if cumulative_confidences else np.exp(log_prob_sum_full)

    return P_T_given_I_Q_full, entropy_sum, cumulative_confidences

#### Recursion Methods ####
def layer_mean_based_recursion(attn = None, attn_threshold = 0.1 , image_mask = None):
    mask = (attn >= attn_threshold).to(torch.float16)
    
    return mask

def layer_mean_topk_based_recursion(attn = None, top_k = 0.1, image_mask = None):
    flattened_attn = attn.view(-1) 
    flattened_attn = flattened_attn.float()

    threshold_index = int(len(flattened_attn) * (top_k)) 
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    image_mask = (attn >= threshold_value).to(torch.float16)
    
    return image_mask

def confidence_topk_based_recursion(attn = None, top_k = 0.1, sequences = None, scores= None, image_mask = None): 
    _, _, cumulative_confidences = calculate_entropy_and_all_confidences(sequence = sequences , scores = scores)
                                                                         
    calculated_threshold = confidence_based_threshold(cumulative_confidences= cumulative_confidences, base_threshold=top_k)
    flattened_attn = attn.view(-1).float()
    threshold_index = int(len(flattened_attn) * (calculated_threshold))
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] >= threshold_value:
                image_mask[row][col] = 1
    
    return image_mask

def calculate_attention_entropy(attn):
    """
    Calculate the entropy of the attention map.
    Args:
        attn (torch.Tensor): Attention map of shape (num_heads, seq_len, seq_len).
    Returns:
        float: Mean entropy across all attention heads.
    """
    # Normalize attention map along the last dimension (softmax-like normalization)
    attn_probs = attn / attn.sum(dim=-1, keepdim=True)

    # Avoid log(0) by adding a small epsilon
    eps = 1e-9
    attn_probs = attn_probs + eps

    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)

    # Average entropy across all heads
    mean_entropy = entropy.mean().item()
    return mean_entropy


def attn_entropy_topk_based_recursion(attn=None, image_mask=None, base_top_k=0.3, return_diagnostics=False):
    """
    Adjust Top-K threshold dynamically based on attention map entropy and record the entropy.
    """
    if return_diagnostics and not torch.isfinite(attn).all():
        raise ValueError("attention values must be finite for audit diagnostics")

    def calculate_attention_entropy(attn):
        """
        Calculate the entropy of the attention map.
        """
        # Normalize attention map along the last dimension (softmax-like normalization)
        attn_probs = attn / attn.sum(dim=-1, keepdim=True)

        # Avoid log(0) by adding a small epsilon
        eps = 1e-9
        attn_probs = attn_probs + eps

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)

        # Average entropy across all heads
        return entropy.mean().item()

    # Calculate entropy of the attention map
    entropy = calculate_attention_entropy(attn)

    # Dynamically adjust the threshold based on entropy
    max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float))  # Max entropy for uniform dist
    normalized_entropy = entropy / max_entropy.item()
    calculated_threshold = (np.exp(base_top_k * normalized_entropy)-1) / (np.exp(base_top_k)-1)
    calculated_threshold = max(min(calculated_threshold, 1.0),0.1)

    # Flatten attention map and calculate Top-K threshold
    flattened_attn = attn.view(-1).float()
    threshold_index = int(len(flattened_attn) * calculated_threshold)
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    # Update the image mask
    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] >= threshold_value:
                image_mask[row][col] = 1

    if return_diagnostics:
        flattened_probs = flattened_attn / flattened_attn.sum()
        entropy_2d = -torch.sum(flattened_probs * torch.log(flattened_probs + 1e-9)).item()
        selected_token_count = int(image_mask.sum().item())
        total_token_count = int(image_mask.numel())

        diagnostics = {
            "second_entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "entropy_2d": float(entropy_2d),
            "attention_cutoff": float(threshold_value.item()),
            "requested_keep_ratio": float(calculated_threshold),
            "actual_keep_ratio": selected_token_count / total_token_count,
            "selected_token_count": selected_token_count,
            "total_token_count": total_token_count,
            "selection_grid_shape": [int(size) for size in attn.shape],
        }
        if not all(
            np.isfinite(value)
            for key, value in diagnostics.items()
            if key not in {
                "selected_token_count",
                "total_token_count",
                "selection_grid_shape",
            }
        ):
            raise ValueError("selector diagnostics must be finite")
        return image_mask, calculated_threshold, diagnostics

    return image_mask, calculated_threshold

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


class ContrastiveDecoder:
    def __init__(self, cd_strategy, eos_token_id=2, pad_token_id=2):
        self.cd_strategy = cd_strategy
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def compute_final_logits(self, stage_logit_list, contrastive_alphas, attention_maps=None, cutoff=True):
        final_logit = stage_logit_list[-1].clone()  # Start with the last stage logits
        cutoff_top_k = False
        cutoff_top_p = False

        if final_logit.argmax(dim=-1).item() in {self.eos_token_id, self.pad_token_id}:
            return final_logit

        for idx in range(len(stage_logit_list) - 1):
            p_v = F.softmax(stage_logit_list[idx + 1], dim=-1)
            p_d = F.softmax(stage_logit_list[idx], dim=-1)

            if self.cd_strategy == "code_adaptive":
                cutoff = False
                kld_alpha = self.kl_adaptive_weight(p_v, p_d, contrastive_alphas[idx])
                cutoff_value = kld_alpha * p_v.max(dim=-1, keepdim=True).values
                diffs = contrastive_alphas[idx] * ((1 + kld_alpha) * stage_logit_list[idx + 1] - kld_alpha * stage_logit_list[idx])
                cd_logits = diffs.masked_fill(p_v < cutoff_value, -float("inf"))

            elif self.cd_strategy == "entropy":
                weight = self.entropy_weight(p_v, p_d)
                cd_logits = contrastive_alphas[idx] * (weight * stage_logit_list[idx + 1] - (1 - weight) * stage_logit_list[idx])

            elif self.cd_strategy == "attention" and attention_maps is not None:
                attn_v = attention_maps[idx + 1]
                attn_d = attention_maps[idx]
                weight_v, weight_d = self.attention_sharpness_weight(attn_v, attn_d)
                cd_logits = contrastive_alphas[idx] * (weight_v * stage_logit_list[idx + 1] - weight_d * stage_logit_list[idx])

            elif self.cd_strategy == "jensen_shannon":
                weight = self.jensen_shannon_divergence(p_v, p_d)
                cd_logits = contrastive_alphas[idx] * (weight * stage_logit_list[idx + 1] - (1 - weight) * stage_logit_list[idx])

            elif self.cd_strategy == "wasserstein":
                weight = self.wasserstein_distance(p_v, p_d)
                cd_logits = contrastive_alphas[idx] * (weight * stage_logit_list[idx + 1] - (1 - weight) * stage_logit_list[idx])

            elif self.cd_strategy == "hellinger":
                weight = self.hellinger_distance(p_v, p_d)
                cd_logits = contrastive_alphas[idx] * (weight * stage_logit_list[idx + 1] - (1 - weight) * stage_logit_list[idx])

            elif self.cd_strategy == "euclidean":
                weight = self.euclidean_distance(p_v, p_d)
                cd_logits = contrastive_alphas[idx] * (weight * stage_logit_list[idx + 1] - (1 - weight) * stage_logit_list[idx])
            
            elif self.cd_strategy == "confidence":
                confidence_weight = p_v.max(dim=-1, keepdim=True).values / (p_v.max(dim=-1, keepdim=True).values + p_d.max(dim=-1, keepdim=True).values)
                cd_logits = contrastive_alphas[idx] * (confidence_weight * stage_logit_list[idx + 1] - (1 - confidence_weight) * stage_logit_list[idx])

            elif self.cd_strategy == "cutoff_top_k":
                cutoff_top_k = True
                cd_logits = contrastive_alphas[idx] * (stage_logit_list[idx + 1] - stage_logit_list[idx])

            elif self.cd_strategy == "cutoff_top_p":
                cutoff_top_k = True
                cd_logits = contrastive_alphas[idx] * (stage_logit_list[idx + 1] - stage_logit_list[idx])
            
            elif self.cd_strategy == "diff_top_k":
                cutoff_top_k = True
                sorted_logits, _  = torch.sort(stage_logit_list[-1].squeeze(0), descending=True)
                first_place = sorted_logits[0].item()
                second_place = sorted_logits[1].item()
                difference = first_place - second_place
                weight = 1 / (difference + 1e-8)
                cd_logits = contrastive_alphas[idx] * (stage_logit_list[idx + 1] - stage_logit_list[idx]) * weight

            else:
                cd_logits = contrastive_alphas[idx] * (stage_logit_list[idx + 1] - stage_logit_list[idx])

            # Update final logits
            final_logit += cd_logits

        # Apply cutoff for the last stage logits if cutoff is specified
        if cutoff:
            last_stage_logit = stage_logit_list[-1].clone()
            if cutoff_top_k:
                k = 20                
                top_k_values, top_k_indices = last_stage_logit.topk(k, dim=-1)
                print(f"indices: {top_k_indices}, values: {top_k_values}")
                cutoff_threshold = top_k_values[:, -1].unsqueeze(-1)  # Smallest value in the top-k                
            else:
                cutoff_coeff = 0.2                
                cutoff_threshold = cutoff_coeff * last_stage_logit.max(dim=-1, keepdim=True).values
            
            final_logit = final_logit.masked_fill(last_stage_logit < cutoff_threshold, -float("inf"))

        return final_logit

    @staticmethod
    def kl_adaptive_weight(p_v, p_d, cd_alpha):
        kl_d = 0.5 * ((torch.log2(torch.abs(p_v - p_d) ** cd_alpha + 1)) * (p_v + p_d)).sum(dim=-1).unsqueeze(-1)
        return 1 - kl_d

    @staticmethod
    def entropy_weight(p_v, p_d):
        entropy_v = -(p_v * p_v.log()).sum(dim=-1)
        entropy_d = -(p_d * p_d.log()).sum(dim=-1)
        return (1 / entropy_v) / ((1 / entropy_v) + (1 / entropy_d))

    @staticmethod
    def attention_sharpness_weight(attn_v, attn_d):
        attn_sharpness_v = 1 / (attn_v.var(dim=-1, keepdim=True) + 1e-6)
        attn_sharpness_d = 1 / (attn_d.var(dim=-1, keepdim=True) + 1e-6)
        total_sharpness = attn_sharpness_v + attn_sharpness_d
        return attn_sharpness_v / total_sharpness, attn_sharpness_d / total_sharpness

    @staticmethod
    def jensen_shannon_divergence(p_v, p_d):
        m = 0.5 * (p_v + p_d)
        kl_v = (p_v * (p_v / m).log()).sum(dim=-1)
        kl_d = (p_d * (p_d / m).log()).sum(dim=-1)
        js_div = 0.5 * kl_v + 0.5 * kl_d
        return 1 - js_div

    @staticmethod
    def wasserstein_distance(p_v, p_d):
        return 1 / (1 + torch.cdist(p_v, p_d, p=1).mean())

    @staticmethod
    def hellinger_distance(p_v, p_d):
        return 1 / (1 + torch.sqrt(torch.sum((torch.sqrt(p_v) - torch.sqrt(p_d)) ** 2, dim=-1)) / torch.sqrt(torch.tensor(2.0)))

    @staticmethod
    def euclidean_distance(p_v, p_d):
        return 1 / (1 + torch.sqrt(torch.sum((p_v - p_d) ** 2, dim=-1)))
