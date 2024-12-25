import gc
import torch
import tqdm
import transformers
from metrics import get_roc_metrics, get_precision_recall_metrics


def get_experiment_results(predictions, name):
    fpr, tpr, roc_auc = get_roc_metrics(predictions["orig"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(predictions["orig"], predictions["samples"])
    print(f"ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return dict(
        name=name,
        predictions=predictions,
        roc_auc=roc_auc,
        fpr=fpr,
        tpr=tpr,
        pr_auc=pr_auc,
        precision=p,
        recall=r
    )


def run_threshold_experiment(criterion_fn, name, data, n_samples=200, batch_size=50):
    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name}"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]
        for idx in range(len(original_text)):
            results.append({
                "original": criterion_fn(original_text[idx]),
                "sampled": criterion_fn(sampled_text[idx]),
            })
    predictions = {
        "orig": [x["original"] for x in results],
        "samples": [x["sampled"] for x in results],
    }
    return get_experiment_results(predictions, f"{name}_threshold")


def run_supervised_experiment(model, data, name, device="cpu", n_samples=200, batch_size=50):
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    orig, sampled = data["original"], data["sampled"]
    with torch.no_grad():
        orig_preds = []
        for batch in tqdm.tqdm(range(n_samples // batch_size), desc="Evaluating orig"):
            batch_orig = orig[batch * batch_size:(batch + 1) * batch_size]
            batch_orig = tokenizer(batch_orig, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            orig_preds.extend(detector(**batch_orig).logits.softmax(-1)[:,0].tolist())
        sampled_preds = []
        for batch in tqdm.tqdm(range(n_samples // batch_size), desc="Evaluating sampled"):
            batch_sampled = sampled[batch * batch_size:(batch + 1) * batch_size]
            batch_sampled = tokenizer(batch_sampled, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            sampled_preds.extend(detector(**batch_sampled).logits.softmax(-1)[:,0].tolist())
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    predictions = {
        "orig": orig_preds,
        "samples": sampled_preds,
    }
    return get_experiment_results(predictions, f"{name}_{model}_supervised")


def run_perturbation_experiment(perturbation_stats, name):
    predictions = {"orig": [], "samples": []}
    for res in perturbation_stats:
        predictions["orig"].append(res["orig_ll"] - res["p_orig_ll"])
        predictions["samples"].append(res["sampled_ll"] - res["p_sampled_ll"])
    return get_experiment_results(predictions, f"{name}_perturbations")
