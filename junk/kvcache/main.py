import time
from contextlib import contextmanager


@contextmanager
def measure(step: str):
    print(f"Measuring {step}")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{step} took {end - start:.2f} seconds")


with measure("Imports"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from generate import generate, prebake_system_prompt

with measure("Initialization"):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", load_in_4bit=True
    ).to("cuda")
    book = tokenizer.encode("book")[0]

messages = [
    {"role": "system", "content": "Obey the user. " * 2000},
]

with measure("Baking"):
    system_cache = prebake_system_prompt(
        model, tokenizer, system_content="You are a helpful AI assistant."
    )


def compose(*funcs):
    def mp(logits: torch.Tensor, step: int, generated_ids: list[int]) -> torch.Tensor:
        for func in funcs:
            logits = func(logits, step, generated_ids)
        return logits
    return mp


def create_no_stop_tweak(tokenizer, stop_token_ids=None):
    """
    Creates a tweak function that prevents generation from stopping.

    Args:
        tokenizer: The tokenizer to get the EOS token ID from
        stop_token_ids: Optional list of token IDs to block. Defaults to [eos_token_id]
    """
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id]

    def no_stop_tweak(
        logits: torch.Tensor, step: int, generated_ids: list[int]
    ) -> torch.Tensor:
        """Set probability of stop tokens to zero."""
        for token_id in stop_token_ids:
            logits[0, token_id] = float("-inf")
        return logits

    def uplift_book(
        logits: torch.Tensor, step: int, generated_ids: list[int]
    ) -> torch.Tensor:
        logits[0, book] *= 100
        return logits
    return compose(uplift_book)


with measure("Generation"):
    while (i := input(">>> ")).strip() != "!quit":
        initial = time.perf_counter()
        first = True
        for token in generate(
            model,
            tokenizer,
            prompt=i,
            max_new_tokens=10000,
            temperature=1.4,
            tweak_logits=create_no_stop_tweak(tokenizer),
            prebaked_system=system_cache,
            stream=True,
        ):
            assert not isinstance(token, str)
            if first:
                print(
                    f"TFT (Time to first token): {time.perf_counter() - initial:.2f} seconds"
                )
                first = False
            print(token["text"], end="", flush=True)
        print("\n---")

# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
