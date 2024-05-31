import random
from collections import namedtuple

import torch

#########
# model #
#########
def perturb_attn_map_llama(module, input, kwargs, output, inv_temp: float=1.0, guidance_scale: float=7.5):
    hidden_states = kwargs['hidden_states']
    attention_mask = kwargs['attention_mask']
    position_ids = kwargs['position_ids']
    past_key_value = kwargs['past_key_value']
    output_attentions = kwargs['output_attentions']
    use_cache = kwargs['use_cache']
    cache_position = kwargs['cache_position']

    if output_attentions:
        raise NotImplementedError()

    bsz, q_len, _ = hidden_states.size()

    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

    cos, sin = module.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # In case static cache is used, it is an instance attribute.
    past_key_value = getattr(module, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    # original attn
    k_len = key_states.size(-2)
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :k_len]

    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output_o = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=module.attention_dropout if module.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
    )

    attn_output_o = attn_output_o.transpose(1, 2).contiguous()
    attn_output_o = attn_output_o.view(bsz, q_len, module.hidden_size)
    attn_output_o = module.o_proj(attn_output_o)

    # perturbed attn
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output_p = torch.nn.functional.scaled_dot_product_attention(
        inv_temp * query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=module.attention_dropout if module.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
    )

    attn_output_p = attn_output_p.transpose(1, 2).contiguous()
    attn_output_p = attn_output_p.view(bsz, q_len, module.hidden_size)
    attn_output_p = module.o_proj(attn_output_p)

    # guidance
    attn_output = attn_output_o + guidance_scale * (attn_output_o - attn_output_p)

    return attn_output, None, past_key_value

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_module_by_name(module, total_module_name):
    for module_name in total_module_name.split('.'):
        if module_name.isnumeric():
            module = module[int(module_name)]
        else:
            module = getattr(module, module_name)
            setattr(module, 'module_name', total_module_name)
    return module


#############
# Eval-GPQA #
#############
choices = ['(A)', '(B)', '(C)', '(D)']

def get_logit_idx(pipe):
    choices_logit_idx_list = []
    for choice in choices:
        logit_idx = pipe.tokenizer.encode(choice)
        assert len(logit_idx) == 3
        choices_logit_idx_list.append(logit_idx[1])
    return choices_logit_idx_list

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def process_gpaq(ds):
    return [shuffle_choices_and_create_example(row) for row in ds['train']]

def shuffle_choices_and_create_example(row) -> Example:
    list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
    random.shuffle(list_choices)
    example = Example(row['Question'], list_choices[0], list_choices[1], list_choices[2], list_choices[3], list_choices.index(row['Correct Answer']))
    return example

def zero_shot_prompt(example) -> str:
    """Creates a zero-shot prompt given a single example. Uses the prompt format from this paper on Scalable Oversight: 
    https://arxiv.org/abs/2211.03540"""
    prompt = base_prompt(example)
    prompt += f"\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""
    return prompt

def base_prompt(example) -> str:
    """Creates a zero-shot prompt given a single example. Uses the prompt format from this paper on Scalable Oversight: 
    https://arxiv.org/abs/2211.03540"""
    prompt = f"What is the correct answer to this question: {example.question}"
    prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    return prompt

llama_layer_list = [
    'model.layers.0.self_attn',
    'model.layers.1.self_attn',
    'model.layers.2.self_attn',
    'model.layers.3.self_attn',
    'model.layers.4.self_attn',
    'model.layers.5.self_attn',
    'model.layers.6.self_attn',
    'model.layers.7.self_attn',
    'model.layers.8.self_attn',
    'model.layers.9.self_attn',
    'model.layers.10.self_attn',
    'model.layers.11.self_attn',
    'model.layers.12.self_attn',
    'model.layers.13.self_attn',
    'model.layers.14.self_attn',
    'model.layers.15.self_attn',
    'model.layers.16.self_attn',
    'model.layers.17.self_attn',
    'model.layers.18.self_attn',
    'model.layers.19.self_attn',
    'model.layers.20.self_attn',
    'model.layers.21.self_attn',
    'model.layers.22.self_attn',
    'model.layers.23.self_attn',
    'model.layers.24.self_attn',
    'model.layers.25.self_attn',
    'model.layers.26.self_attn',
    'model.layers.27.self_attn',
    'model.layers.28.self_attn',
    'model.layers.29.self_attn',
    'model.layers.30.self_attn',
    'model.layers.31.self_attn',
]


# # get GPT module name list
# MODULE_LIST = []
# def get_all_module_name_list(module, current_module_name):
#     global MODULE_LIST
#     if module.__class__.__name__ == 'GPT2Attention':
#         MODULE_LIST.append('.'.join(current_module_name))
#     elif module.__class__.__name__ == 'GPT2MLP':
#         MODULE_LIST.append('.'.join(current_module_name))
#     else:
#         for k in module._modules.keys():
#             get_all_module_name_list(getattr(module, k), current_module_name + [k])

# # get LLAMA module name list
# MODULE_LIST = []
# def get_all_module_name_list(module, current_module_name):
#     global MODULE_LIST
#     if module.__class__.__name__ == 'LlamaSdpaAttention':
#         MODULE_LIST.append('.'.join(current_module_name))
#     else:
#         for k in module._modules.keys():
#             get_all_module_name_list(getattr(module, k), current_module_name + [k])


