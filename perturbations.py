import numpy as np
import re
import tqdm


def tokenize_and_mask(text, span_length=10, pct=0.3):
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"
    n_spans = int(pct * len(tokens) / (span_length + 2))
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - 2)
        search_end = min(len(tokens), end + 2)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    text = " ".join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


def replace_masks(model, tokenizer, texts, device="cpu"):
    n_expected = count_masks(texts)
    stop_id = tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**tokens, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop_id)
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def get_perturbed_texts(model, tokenizer, masked_texts, device="cpu"):
    pattern = re.compile(r"<extra_id_\d+>")
    raw_fills = replace_masks(model, tokenizer, masked_texts, device=device)
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in raw_fills]
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    tokens = [x.split(" ") for x in masked_texts]
    n_expected = count_masks(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
    perturbed_texts = [" ".join(x) for x in tokens]
    return perturbed_texts


def perturb_texts(model, tokenizer, texts, device="cpu", span_length=10, pct=0.3):
    masked_texts = [tokenize_and_mask(x, span_length, pct) for x in texts]
    perturbed_texts = get_perturbed_texts(model, tokenizer, masked_texts, device=device)
    while "" in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
        masked_texts = [tokenize_and_mask(x, span_length, pct) for idx, x in enumerate(texts) if idx in idxs]
        new_perturbed_texts = get_perturbed_texts(model, tokenizer, masked_texts, device=device)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
    return perturbed_texts


def perturb_texts_chunked(model, tokenizer, texts, device="cpu", chunk_size=20, span_length=10, pct=0.3):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts(model, tokenizer, texts[i:i + chunk_size], device, span_length, pct))
    return outputs


def get_perturbation_stats(perturb_fn, ll_fn, data, n_perturbations=10):
    original_text = data["original"]
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    sampled_text = data["sampled"]
    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    res = []
    for idx in tqdm.tqdm(range(len(original_text)), desc="Computing perturbation stats"):
        orig_ll = ll_fn(original_text[idx])
        perturbed_orig_ll = np.mean(list(map(ll_fn, p_original_text[idx*n_perturbations:(idx+1)*n_perturbations])))
        sampled_ll = ll_fn(sampled_text[idx])
        perturbed_sampled_ll = np.mean(list(map(ll_fn, p_sampled_text[idx*n_perturbations:(idx+1)*n_perturbations])))
        res.append({"orig_ll": orig_ll, "p_orig_ll": perturbed_orig_ll, "sampled_ll": sampled_ll, "p_sampled_ll": perturbed_sampled_ll})
    return res
