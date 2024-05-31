
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face에서 사전학습된 LLaMA 모델과 토크나이저 불러오기
model = AutoModelForCausalLM.from_pretrained("gpt-2")
tokenizer = AutoTokenizer.from_pretrained("gpt-2")

def generate_with_pag(prompt, guidance_scale=1.0, perturb_attn_ids=None):
    # 입력 prompt를 토크나이즈하여 input_ids 생성
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # perturb_attn_ids가 None이면 빈 리스트로 초기화
    if perturb_attn_ids is None:
        perturb_attn_ids = []

    # hidden states를 출력하기 위해 output_hidden_states=True로 설정
    model_kwargs = {"output_hidden_states": True}
    outputs = model(input_ids, **model_kwargs)
    hidden_states = outputs.hidden_states

    # 최종 hidden state
    final_hidden_state = hidden_states[-1]

    # Perturbed self-attention을 적용한 최종 hidden state 계산
    perturbed_final_hidden_state = final_hidden_state.clone()
    for layer_idx in range(model.config.num_hidden_layers):
        # 현재 layer의 attention이 perturb_attn_ids에 포함되어 있는지 확인
        if layer_idx in perturb_attn_ids:
            # 현재 layer의 self-attention에 identity matrix를 attention mask로 적용
            attn_mask = torch.eye(input_ids.size(-1), dtype=torch.bool).to(input_ids.device)
            perturbed_hidden_state = model.model.layers[layer_idx].self_attn(
                hidden_states[layer_idx], attn_mask=attn_mask
            )[0]
        else:
            # PAG를 적용하지 않는 layer는 원래의 hidden state를 사용
            perturbed_hidden_state = hidden_states[layer_idx]
        
        # Perturbed hidden state를 다음 layer에 전파
        for i in range(layer_idx + 1, model.config.num_hidden_layers):
            perturbed_hidden_state = model.model.layers[i](perturbed_hidden_state)[0]
        
        # 최종 perturbed hidden state 업데이트
        perturbed_final_hidden_state = perturbed_hidden_state

    # 최종 hidden state를 language model head에 통과시켜 next token의 logit 계산
    logits_original = model.lm_head(final_hidden_state)[:, -1, :]
    logits_perturbed = model.lm_head(perturbed_final_hidden_state)[:, -1, :]

    # PAG 적용
    logits = logits_original + guidance_scale * (logits_original - logits_perturbed)

    # logit이 가장 큰 token을 선택
    next_token_id = torch.argmax(logits, dim=-1)

    # 선택된 token을 decode하여 반환
    return tokenizer.decode(next_token_id)


def main():
    prompt = 'Hello, my name is'
    next_token = generate_with_pag(prompt, guidance_scale=2.0, perturb_attn_ids=[3, 7, 11])
    print(f'Prompt: {prompt}')
    print(f'Next token: {next_token}')



if '__main__' == __name__:
    main()