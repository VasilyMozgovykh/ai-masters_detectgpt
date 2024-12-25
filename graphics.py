import matplotlib.pyplot as plt
import os


def plot_llr_hist(perturbation_stats, name="", img_folder="./"):
    plt.clf()
    predictions = {"orig": [], "samples": []}
    for res in perturbation_stats:
        predictions["orig"].append(res["orig_ll"] - res["p_orig_ll"])
        predictions["samples"].append(res["sampled_ll"] - res["p_sampled_ll"])
    plt.figure(figsize=(7, 5))
    plt.hist(predictions["orig"], alpha=0.5, bins='auto', label='original')
    plt.hist(predictions["samples"], alpha=0.5, bins='auto', label='sampled')
    plt.xlabel("Perturbation discrepancy")
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    if name:
        plt.title(name)
        plt.savefig(os.path.join(img_folder, f"{name}_llr_hist.png"))
    plt.show()


def plot_roc_curves(experiments, name="", img_folder="./"):
    plt.clf()
    plt.figure(figsize=(7, 5))
    for exp in experiments:
        plt.plot(exp["fpr"], exp["tpr"], label=f"{exp['name']}, roc_auc={exp['roc_auc']:.3f}")
        print(f"{exp['name']} roc_auc: {exp['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right", fontsize=6)
    if name:
        plt.title(f"ROC {name}")
        plt.savefig(os.path.join(img_folder, f"{name}_roc.png"))
    plt.show()
