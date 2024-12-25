import datasets
import random
import transformers
import tqdm


def trim_to_shorter_length(text1, text2):
    shorter_length = min(len(text1.split(" ")), len(text2.split(" ")))
    text1 = " ".join(text1.split(" ")[:shorter_length])
    text2 = " ".join(text2.split(" ")[:shorter_length])
    return text1, text2


def sample_from_model(model, tokenizer, texts, device="cpu", min_words=55, prompt_tokens=30):
    all_encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}
    decoded = ["" for _ in range(len(texts))]
    while min(len(x.split()) for x in decoded) < min_words:
        sampling_kwargs = {"top_p": 0.96, "top_k": 40}
        outputs = model.generate(**all_encoded, min_length=150, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def generate_samples(model, tokenizer, raw_data, device="cpu", batch_size=50, min_words=55):
    data = {
        "original": [],
        "sampled": [],
    }
    for batch in tqdm.tqdm(range(len(raw_data) // batch_size), desc="Sampling from base_model"):
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(model, tokenizer, original_text, device=device, min_words=min_words)
        for o, s in zip(original_text, sampled_text):
            o, s = trim_to_shorter_length(o, s)
            data["original"].append(o)
            data["sampled"].append(s)
    return data


def prepare_data(dataset_name, dataset_col, model, tokenizer, device, dataset_size=5_000, n_samples=200):
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
    data = datasets.load_dataset(dataset_name, split="train")[dataset_col]
    data = [" ".join(x.strip().split()) for x in set(data)]
    long_data = [x for x in data if len(x.split()) > 250]
    if len(long_data) > 0:
        data = long_data
    random.shuffle(data)
    data = data[:dataset_size]
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    return generate_samples(model, tokenizer, data[:n_samples], device=device)
