import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# === Configuration ===
MODEL_NAME = "tiiuae/falcon-7b-instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model & tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    local_files_only=False
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=False
)

# Falcon’s context length
MAX_CONTEXT = tokenizer.model_max_length  # typically 2048

# === Helper: greedy next-token ===
def greedy_next_token(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)

# === Inference + injection loop ===
def inference_with_injection(original_prompt: str,
                             inject_prompt: str,
                             pre_tokens: int,
                             post_tokens: int) -> str:
    # Tokenize original prompt
    enc = tokenizer(original_prompt, return_tensors="pt").to(DEVICE)
    generated = enc.input_ids
    # Run initial pass to get past_key_values
    out = model(**enc, use_cache=True)
    past = DynamicCache.from_legacy_cache(out.past_key_values)
    eos_id = tokenizer.eos_token_id

    # 1) Generate pre_tokens tokens greedily
    for _ in range(pre_tokens):
        if generated.shape[1] >= MAX_CONTEXT:
            break
        last = generated[:, -1].unsqueeze(-1)
        out = model(input_ids=last, past_key_values=past, use_cache=True)
        past = DynamicCache.from_legacy_cache(out.past_key_values)
        nxt = greedy_next_token(out.logits[:, -1, :])
        generated = torch.cat([generated, nxt], dim=1)

    # 2) Inject new prompt (skip BOS token)
    inj_ids = tokenizer(inject_prompt, return_tensors="pt").input_ids.to(DEVICE)[0, 1:]
    for tid in inj_ids:
        tid = tid.view(1, 1)
        out = model(input_ids=tid, past_key_values=past, use_cache=True)
        past = DynamicCache.from_legacy_cache(out.past_key_values)
        generated = torch.cat([generated, tid], dim=1)

    # 3) Generate post_tokens tokens greedily
    for _ in range(post_tokens):
        if generated.shape[1] >= MAX_CONTEXT:
            break
        last = generated[:, -1].unsqueeze(-1)
        out = model(input_ids=last, past_key_values=past, use_cache=True)
        past = DynamicCache.from_legacy_cache(out.past_key_values)
        nxt = greedy_next_token(out.logits[:, -1, :])
        generated = torch.cat([generated, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break

    # Decode and return full text
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# === Usage ===
if __name__ == "__main__":
    max_pre = 5   # tokens before injection
    max_post = 50  # tokens after injection

    original_q = "Q: Johnny has 5 apples. Margaret has 3 more than him. How many apples does Margaret have?\nA:"
    inject_q   = "IMPORT CONTEXT CHANGE: Actually, Johnny has 4 apples, not 5. If Margret has 2 more apples than current Johnny how many apples does she have?\nA:"

    result = inference_with_injection(original_q, inject_q, max_pre, max_post)
    print(result)
    with open("output.txt", "w") as f:
        f.write(result)