import json
from functools import partial

import numpy as np
import transformers
import torch
import huggingface_hub
from datasets import load_dataset

huggingface_hub.login("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")

from pag.utils import (
    get_module_by_name,
    perturb_attn_map_llama,
    get_logit_idx,
    base_prompt,
    process_gpaq,
    llama_layer_list,
)

pipe = transformers.pipeline(
    "text-generation", model="meta-llama/Meta-Llama-3-8B", 
    model_kwargs={"torch_dtype": torch.float16}, device_map="cuda"
)

@torch.no_grad()
def get_acc(pipe, examples, exp_name:str):
    correct = 0
    for example in tqdm(examples, desc=exp_name):
        prompt = base_prompt(example)
        inputs = pipe.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = pipe.model(**inputs)
            
        correct += (
            outputs.logits[0, -1, choices_logit_idx_tensor].argmax().item()
            == example.correct_index
        )
        acc = 100 * correct / len(examples)
    return acc

def save_dict_as_json(data: dict, path):
    with open(path, 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    # track result
    path = 'llama-gpqa.json'
    preformance_dict = {}
    
    # load gpqa dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    examples = process_gpaq(ds)
    choices_logit_idx_tensor = torch.tensor(get_logit_idx(pipe), device='cuda')
    
    # baseline performance
    acc = get_acc(pipe, examples, exp_name='baseline')
    preformance_dict['baseline'] = acc
    save_dict_as_json(preformance_dict, path)
    
    # PAG exp list
    start_layer_idx = 7
    end_layer_idx = 15
    guidance_scale_list = [0.25, 0.5, 0.75, 1.0]
    inv_temp_list = [3/1e1, 1/1e1, 3/1e2, 1/1e2]
    
    hook_layer_list = llama_layer_list[start_layer_idx-1:end_layer_idx]
    
    for guidance_scale in guidance_scale_list:
        for inv_temp in inv_temp_list:
            # save name
            exp_name = (
                'guidance_scale_{}-temp_{}'
                .format(guidance_scale, round(1/inv_temp, 2))
            )
            
            # hook PAG
            handles = []
            for layer_name in hook_layer_list:
                module = get_module_by_name(pipe.model, layer_name)
                perturb_attn_map_fn = partial(perturb_attn_map_llama, guidance_scale=guidance_scale, inv_temp=inv_temp)
                handle = module.register_forward_hook(perturb_attn_map_fn, with_kwargs=True)
                handles.append(handle)
            
            # experiment!
            acc = get_acc(pipe, examples, exp_name=exp_name)
            
            # remove PAG
            for handle in handles:
                handle.remove()
            
            # save results
            preformance_dict[exp_name] = acc
            save_dict_as_json(preformance_dict, path)