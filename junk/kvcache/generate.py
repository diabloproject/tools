import torch
import torch.nn.functional as F
from typing import Callable, Optional, List, Iterator, Dict, Any, Tuple, Union, overload, Literal
from transformers import PreTrainedModel, PreTrainedTokenizer

def prebake_system_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    system_content: str,
    **model_kwargs
) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Pre-compute KV cache for a system prompt using chat template.

    Returns: (past_key_values, attention_mask, system_input_ids)
    """
    messages = [{"role": "system", "content": system_content}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    inputs = tokenizer(chat_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **model_kwargs
        )

    return outputs.past_key_values, attention_mask, input_ids

def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    prebaked_system: Optional[Tuple[Any, torch.Tensor, torch.Tensor]] = None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = True,
    tweak_logits: Optional[Callable[[torch.Tensor, int, List[int]], torch.Tensor]] = None,
    on_token: Optional[Callable[[str, int], None]] = None,
    stop_token_ids: Optional[List[int]] = None,
    stream: bool = False,
    **model_kwargs
) -> Union[Iterator[Dict[str, Any]], str]:
    """
    Generate response for a SINGLE chat sequence with optional system prompt pre-baking.

    Args:
        prompt: User prompt string
        system_prompt: System prompt (ignored if prebaked_system is provided)
        prebaked_system: Tuple from prebake_system_prompt()
    """

    # Build inputs based on whether we have a pre-baked system prompt
    if prebaked_system:
        past_key_values, system_mask, system_ids = prebaked_system

        # Format user message + assistant prompt
        messages = [{"role": "user", "content": prompt}]
        user_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        user_inputs = tokenizer(user_text, return_tensors="pt")
        input_ids = user_inputs["input_ids"].to(model.device)
        user_mask = user_inputs["attention_mask"].to(model.device)

        # Combine masks
        attention_mask = torch.cat([system_mask, user_mask], dim=1)
    else:
        # Build full conversation from scratch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        past_key_values = None

    # Generation loop
    step = 0
    generated_ids = []
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id]

    while step < max_new_tokens:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **model_kwargs
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Apply interventions
        if tweak_logits is not None:
            logits = tweak_logits(logits.clone(), step, generated_ids)

        # Sampling logic
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze() if do_sample else torch.argmax(logits, dim=-1).squeeze()
        token_id = next_token.item()

        if token_id in stop_token_ids:
            break

        generated_ids.append(token_id)

        # Stream output
        if stream or on_token:
            text = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if stream:
                yield {"text": text, "token_id": token_id, "step": step}
            if on_token:
                on_token(text, step)

        # Update for next iteration
        input_ids = next_token.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=model.device)], dim=1)
        step += 1

    if not stream:
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


# Helper functions
def apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], penalty: float = 1.2) -> torch.Tensor:
    """Penalize repeated tokens"""
    if not generated_ids:
        return logits

    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits

def boost_word_probability(
    tokenizer: PreTrainedTokenizer,
    logits: torch.Tensor,
    word: str,
    boost: float = 2.0
) -> torch.Tensor:
    """Boost probability of a specific word"""
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    for token_id in token_ids:
        logits[0, token_id] *= boost
    return logits

def suppress_word(
    tokenizer: PreTrainedTokenizer,
    logits: torch.Tensor,
    word: str,
    factor: float = 0.1
) -> torch.Tensor:
    """Suppress a specific word"""
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    for token_id in token_ids:
        logits[0, token_id] *= factor
    return logits

def enforce_min_length(logits: torch.Tensor, step: int, min_length: int, eos_token_id: int) -> torch.Tensor:
    """Prevent early stopping before min_length"""
    if step < min_length:
        logits[0, eos_token_id] = float('-inf')
    return logits
