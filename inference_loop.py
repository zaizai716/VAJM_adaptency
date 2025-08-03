import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch.nn.functional as F
import json

model_path = "/Users/justinyu/.cache/huggingface/hub/models--internlm--internlm2-math-base-7b/snapshots/e4fc5d0416940723dcfd646288a7fa289cb879f4"

# load in model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16, # use float16 for GPU efficiency
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# automatically use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

# captures top 10 logits and adds stage info (pre/post injection)
def print_top10_logits(probs, chosen_token, tokenizer, stage="pre", sampled_prob=None):

    distributions = []
    top_10_logits = []

    # .squeeze() in order to get rid of unnecessary dimensions, make it a 1d tensor
    chosen_tok = tokenizer.decode(chosen_token.squeeze().tolist()) 
    # .tolist() to turn tensor into python list, since tokenizer.decode() only accepts python lists
    chosen_tok_id = chosen_token.item()

    # get top 10 token probabilities from raw logits
    top10_logits_raw, top10_ids = torch.topk(probs, k=10)
    probs_topk = F.softmax(top10_logits_raw[0], dim=-1)

    for prob, token in zip(probs_topk, top10_ids[0]):
        tok_prob = prob.item()
        tok_id = token.item()
        tok_str = tokenizer.decode([tok_id])

        top_10_logits.append({
            "token_id": tok_id,
            "token_string": tok_str,
            "token_probability": tok_prob,
        })

    distributions.append({
        "stage": stage,  # records whether it's pre or post injection
        "chosen_token": chosen_tok,
        "chosen_token_id": chosen_tok_id,
        "sampled_token_probability": sampled_prob, 
        "top_10_logits": top_10_logits,  # ✅ included properly here
    })

    return distributions[0]  # return single dict

# sampling logic with top-k and temperature
# temp < 1 means less randomness
# top-k filters out low-probability junk

def sample_next_token(logits, temperature=0.5, top_k=50):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # top-k sampling
    topk_probs, topk_indices = torch.topk(probs[0], top_k)
    topk_probs = topk_probs / topk_probs.sum()

    sampled_idx = torch.multinomial(topk_probs, num_samples=1).item()
    sampled_token = topk_indices[sampled_idx].item()
    sampled_prob = topk_probs[sampled_idx].item()

    next_token = torch.tensor([[sampled_token]], device=logits.device)
    return next_token, sampled_prob

def inference_loop(prompt, changed_prompt, max_tokens):      

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # feed prompt to model to start, processes prompt
    output = model(**inputs, use_cache=True) # allows model to use internal cache, doesn't forget past text
    # get state so u don't reset internal cache later on
    past = DynamicCache.from_legacy_cache(output.past_key_values)

    generated_ids = inputs["input_ids"]

    # holds all distributions (pre and post injection)
    distributions = []
    eos_token = tokenizer.eos_token_id

    for token in range(max_tokens):

        with torch.no_grad():
            if generated_ids.shape[1] >= model.config.max_position_embeddings:
                print("❌ Reached max context length, stopping generation.")
                break

            last_token = generated_ids[:, -1].unsqueeze(-1) # gets the last token, then adds a dimension, which model expects
            output = model(input_ids=last_token, past_key_values=past, use_cache=True)
            past = DynamicCache.from_legacy_cache(output.past_key_values) # updates state
            next_token, token_prob = sample_next_token(output.logits[:, -1, :])

            # pass in top 10 logits to figure out how distribution changes pre-injection vs post-injection
            probs = torch.softmax(output.logits[:, -1, :], dim=-1) # dim=-1 means to just operate on the LAST dimension
            distribution = print_top10_logits(probs, next_token, tokenizer, stage="pre", sampled_prob=token_prob)
            distributions.append(distribution)

            # torch.cat([]) expects two 2d tensors to concatenate, otherwise will throw error
            generated_ids = torch.cat([generated_ids, next_token], dim=1) # adding that newly generated token to the list
            # don't need to unsqueeze next_token because it's already a 2d tensor

            if token == max_tokens - 1: # occurs when tokens hit max limit, and injection needs to be done
                inject_ids = tokenizer(changed_prompt, return_tensors="pt")["input_ids"].to(device)[:, 1:] # this slicing removes all <BOS> tokens, unnecessary

                # feeds new injected prompt token by token, so model doesn't reprompt entirely
                for token_id in inject_ids[0]:
                    token_id = token_id.view(1, 1) # same as .reshape(), need to make it (1, 1) to match what the model expects
                    output = model(input_ids=token_id, past_key_values=past, use_cache=True)
                    past = DynamicCache.from_legacy_cache(output.past_key_values)
                    generated_ids = torch.cat([generated_ids, token_id], dim=1)

                post_token_count = 0
                while post_token_count < 50: # cap generation after injection
                    post_token_count += 1

                    if generated_ids.shape[1] >= model.config.max_position_embeddings:
                        print("❌ Reached max context length, stopping generation.")
                        break

                    last_token = generated_ids[:, -1].unsqueeze(-1)
                    output = model(input_ids=last_token, past_key_values=past, use_cache=True)
                    past = DynamicCache.from_legacy_cache(output.past_key_values)
                    next_token, token_prob = sample_next_token(output.logits[:, -1, :])

                    probs = torch.softmax(output.logits[:, -1, :], dim=-1) 
                    distribution = print_top10_logits(probs, next_token, tokenizer, stage="post", sampled_prob=token_prob)
                    distributions.append(distribution)

                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    # breaks when next token is <EOS> token, signifying end of model generation
                    if eos_token is not None and next_token.item() == eos_token:
                        break

    # store final model output alongside all distributions
    final_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    full_trace = {
        "final_output": final_output,
        "distributions": distributions  # ✅ this includes top_10_logits!
    }

    with open("full_distribution_trace.json", "w") as f:
        json.dump(full_trace, f, indent=2)

    return final_output

max_tokens = 20
original_prompt = "Q: Johnny has 5 apples. Margaret has 3 more than him. How many apples does Margaret have?\nA:"
changed_prompt = "IMPORT CONTEXT CHANGE: Actually, Johnny gives away 2 apples. How many apples does he have now?\nA:"
inference_loop(original_prompt, changed_prompt, max_tokens)