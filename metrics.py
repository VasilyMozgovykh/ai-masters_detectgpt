import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, auc


@torch.no_grad()
def get_likelihood(model, tokenizer, text, device="cpu"):
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    labels = tokenized.input_ids
    return -model(**tokenized, labels=labels).loss.item()


@torch.no_grad()
def get_rank(model, tokenizer, text, device="cpu", log=False):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        ranks = matches[:,-1] + 1
        if log:
            ranks = torch.log(ranks)
        return ranks.float().mean().item()


@torch.no_grad()
def get_entropy(model, tokenizer, text, device="cpu"):
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**tokenized).logits[:,:-1]
    neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    return -neg_entropy.sum(-1).mean().item()


def get_lls(model, tokenizer, texts, device="cpu"):
    get_ll = lambda text: get_likelihood(model, tokenizer, text, device)
    return list(map(get_ll, texts))


def get_roc_metrics(orig_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(orig_preds) + [1] * len(sample_preds), orig_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(orig_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(orig_preds) + [1] * len(sample_preds), orig_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)
