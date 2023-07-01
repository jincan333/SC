# p_true,  most_likely_probability, second_most+probability, avg_probability
import argparse
import os
import pickle
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from math import isnan
import datetime

from parser_config import add_args, device_map
from utils import set_seed

def ece_curve(predictions, true_labels, n_bins=10):
    bin_edges = np.arange(0., 1.1, 1. / n_bins)
    
    accuracies = []
    confidences = []
    cnt = []
    for k in range(n_bins):
        idx = np.where((predictions >= bin_edges[k]) & (predictions < bin_edges[k + 1]))[0]
        if len(idx) > 0:
            accuracy = accuracy_score(true_labels[idx], np.round(predictions[idx]))
            confidence = np.mean(predictions[idx])
            accuracies.append(accuracy)
            confidences.append(confidence)
            cnt.append(len(idx))
    cnt = [_ / sum(cnt) for _ in cnt]

    return confidences, accuracies, cnt



global args, device, save_path, semantic_tokenizer, semantic_model, bert_model_name, generate_model, generate_tokenizer
parser = argparse.ArgumentParser(description='PyTorch Semantic Experiments')
add_args(parser)
args = parser.parse_args()
print(args)
# Device
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(int(args.gpu))

wandb.init(project='nlg_uncertainty', id=args.run_name, config=args, resume='allow')
run_name = wandb.run.name
os.environ["HF_DATASETS_CACHE"] = args.hf_dataset_cache
# Save Path
save_path = os.path.join(args.save_dir, args.dataset, args.generate_model, args.run_name, str(args.one_time_generate_num)+'_'+str(args.num_generations_per_prompt), 'generation')
# Random seed
set_seed(args.seed)
# Accuracy and confidence file
save_file = os.path.join(save_path, 'generations_ordinary_accuracy_confidence.pkl')
with open(save_file , 'rb') as infile:
    ordinary_acc_conf_list = pickle.load(infile)

most_nan_idx = []
most_ppl_prob = []
most_prob = []
most_norm_ppl_prob = []
most_norm_prob = []
most_exact_match = []
most_rouge1 = []
most_rouge2 = []
most_rougeL = []
most_bertscore = []

sample_nan_idx = []
sample_ppl_prob = []
sample_prob = []
sample_norm_ppl_prob = []
sample_norm_prob = []
sample_exact_match = []
sample_rouge1 = []
sample_rouge2 = []
sample_rougeL = []
sample_bertscore = []

for i, cal in enumerate(ordinary_acc_conf_list):
    if isnan(cal['most_ppl_prob']) or isnan(cal['most_prob']) or isnan(cal['most_norm_ppl_prob']) or isnan(cal['most_norm_prob']):
        most_nan_idx.append(i)
    most_ppl_prob.append(cal['most_ppl_prob'])
    most_prob.append(cal['most_prob'])
    most_norm_ppl_prob.append(cal['most_norm_ppl_prob'])
    most_norm_prob.append(cal['most_norm_prob'])
    most_exact_match.append(cal['most_exact_match'])
    most_rougeL.append(int(cal['most_rougeL'] > 0.3))
    most_bertscore.append(cal['most_bertscore'])

    if isnan(cal['sample_ppl_prob']) or isnan(cal['sample_prob']) or isnan(cal['sample_norm_ppl_prob']) or isnan(cal['sample_norm_prob']):
        sample_nan_idx.append(i)
    sample_ppl_prob.append(cal['sample_ppl_prob'])
    sample_prob.append(cal['sample_prob'])
    sample_norm_ppl_prob.append(cal['sample_norm_ppl_prob'])
    sample_norm_prob.append(cal['sample_norm_prob'])
    sample_exact_match.append(cal['sample_exact_match'])
    sample_rougeL.append(int(cal['sample_rougeL'] > 0.3))
    sample_bertscore.append(cal['sample_bertscore'])
# Drop NaN
most_ppl_prob = [value for index, value in enumerate(most_ppl_prob) if index not in most_nan_idx]
most_prob = [value for index, value in enumerate(most_prob) if index not in most_nan_idx]
most_norm_ppl_prob = [value for index, value in enumerate(most_norm_ppl_prob) if index not in most_nan_idx]
most_norm_prob = [value for index, value in enumerate(most_norm_prob) if index not in most_nan_idx]
most_exact_match = [value for index, value in enumerate(most_exact_match) if index not in most_nan_idx]
most_rougeL = [value for index, value in enumerate(most_rougeL) if index not in most_nan_idx]

sample_ppl_prob = [value for index, value in enumerate(sample_ppl_prob) if index not in sample_nan_idx]
sample_prob = [value for index, value in enumerate(sample_prob) if index not in sample_nan_idx]
sample_norm_ppl_prob = [value for index, value in enumerate(sample_norm_ppl_prob) if index not in sample_nan_idx]
sample_norm_prob = [value for index, value in enumerate(sample_norm_prob) if index not in sample_nan_idx]
sample_exact_match = [value for index, value in enumerate(sample_exact_match) if index not in sample_nan_idx]
sample_rougeL = [value for index, value in enumerate(sample_rougeL) if index not in sample_nan_idx]

# Calculate Brier Score
ppl_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_ppl_prob, most_exact_match)]) / len(most_exact_match)
prob_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_prob, most_exact_match)]) / len(most_exact_match)
norm_ppl_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_norm_ppl_prob, most_exact_match)]) / len(most_exact_match)
norm_prob_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_norm_prob, most_exact_match)]) / len(most_exact_match)
ppl_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_ppl_prob, most_rougeL)]) / len(most_rougeL)
prob_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_prob, most_rougeL)]) / len(most_rougeL)
norm_ppl_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_norm_ppl_prob, most_rougeL)]) / len(most_rougeL)
norm_prob_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(most_norm_prob, most_rougeL)]) / len(most_rougeL)

print('Most Brier Score\n'
      f'ppl_exact_brier: {ppl_exact_brier:.4f}\n'
      f'prob_exact_brier: {prob_exact_brier:.4f}\n'
      f'norm_ppl_exact_brier: {norm_ppl_exact_brier:.4f}\n'
      f'norm_prob_exact_brier: {norm_prob_exact_brier:.4f}\n'
      f'ppl_rouge_brier: {ppl_rouge_brier:.4f}\n'
      f'prob_rouge_brier: {prob_rouge_brier:.4f}\n'
      f'norm_ppl_rouge_brier: {norm_ppl_rouge_brier:.4f}\n'
      f'norm_prob_rouge_brier: {norm_prob_rouge_brier:.4f}\n'
)

ppl_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_ppl_prob, sample_exact_match)]) / len(sample_exact_match)
prob_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_prob, sample_exact_match)]) / len(sample_exact_match)
norm_ppl_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_norm_ppl_prob, sample_exact_match)]) / len(sample_exact_match)
norm_prob_exact_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_norm_prob, sample_exact_match)]) / len(sample_exact_match)
ppl_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_ppl_prob, sample_rougeL)]) / len(sample_rougeL)
prob_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_prob, sample_rougeL)]) / len(sample_rougeL)
norm_ppl_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_norm_ppl_prob, sample_rougeL)]) / len(sample_rougeL)
norm_prob_rouge_brier = sum([(pred - actual) ** 2 for pred, actual in zip(sample_norm_prob, sample_rougeL)]) / len(sample_rougeL)

print('Sample Brier Score\n'
      f'ppl_exact_brier: {ppl_exact_brier:.4f}\n'
      f'prob_exact_brier: {prob_exact_brier:.4f}\n'
      f'norm_ppl_exact_brier: {norm_ppl_exact_brier:.4f}\n'
      f'norm_prob_exact_brier: {norm_prob_exact_brier:.4f}\n'
      f'ppl_rouge_brier: {ppl_rouge_brier:.4f}\n'
      f'prob_rouge_brier: {prob_rouge_brier:.4f}\n'
      f'norm_ppl_rouge_brier: {norm_ppl_rouge_brier:.4f}\n'
      f'norm_prob_rouge_brier: {norm_prob_rouge_brier:.4f}\n'
)

# Plot the ECE curve
# MOST
ppl_exact_conf, ppl_exact_acc, ppl_exact_dens = ece_curve(np.array(most_ppl_prob), np.array(most_exact_match))
prob_exact_conf, prob_exact_acc, prob_exact_dens = ece_curve(np.array(most_prob), np.array(most_exact_match))
norm_ppl_exact_conf, norm_ppl_exact_acc, norm_ppl_exact_dens = ece_curve(np.array(most_norm_ppl_prob), np.array(most_exact_match))
norm_prob_exact_conf, norm_prob_exact_acc, norm_prob_exact_dens = ece_curve(np.array(most_norm_prob), np.array(most_exact_match))
ppl_rougeL_conf, ppl_rougeL_acc, ppl_rougeL_dens = ece_curve(np.array(most_ppl_prob), np.array(most_rougeL))
prob_rougeL_conf, prob_rougeL_acc, prob_rougeL_dens = ece_curve(np.array(most_prob), np.array(most_rougeL))
norm_ppl_rougeL_conf, norm_ppl_rougeL_acc, norm_ppl_rougeL_dens = ece_curve(np.array(most_norm_ppl_prob), np.array(most_rougeL))
norm_prob_rougeL_conf, norm_prob_rougeL_acc, norm_prob_rougeL_dens = ece_curve(np.array(most_norm_prob), np.array(most_rougeL))

image_title = 'MOST_EXACT ECE'
save_file = os.path.join(save_path, image_title)
plt.plot(ppl_exact_conf, ppl_exact_acc, marker='o', label = 'ppl')
plt.plot(prob_exact_conf, prob_exact_acc, marker='.', label = 'prob')
plt.plot(norm_ppl_exact_conf, norm_ppl_exact_acc, marker=',', label = 'norm_ppl')
plt.plot(norm_prob_exact_conf, norm_prob_exact_acc, marker='v', label = 'norm_prob')
plt.plot([0, 1], [0, 1])
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title(image_title)
plt.legend(loc=2)
plt.show()
plt.savefig(save_file+'.png')
plt.close()

image_title = 'MOST_ROUGE ECE'
save_file = os.path.join(save_path, image_title)
plt.plot(ppl_rougeL_conf, ppl_rougeL_acc, marker='o', label = 'ppl')
plt.plot(prob_rougeL_conf, prob_rougeL_acc, marker='.', label = 'prob')
plt.plot(norm_ppl_rougeL_conf, norm_ppl_rougeL_acc, marker=',', label = 'norm_ppl')
plt.plot(norm_prob_rougeL_conf, norm_prob_rougeL_acc, marker='v', label = 'norm_prob')
plt.plot([0, 1], [0, 1])
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title(image_title)
plt.legend(loc=2)
plt.show()
plt.savefig(save_file+'.png')
plt.close()

image_title = 'MOST Density'
save_file = os.path.join(save_path, image_title)
plt.plot(ppl_rougeL_conf, ppl_rougeL_dens, marker='o', label = 'ppl')
plt.plot(prob_rougeL_conf, prob_rougeL_dens, marker='.', label = 'prob')
plt.plot(norm_ppl_rougeL_conf, norm_ppl_rougeL_dens, marker=',', label = 'norm_ppl')
plt.plot(norm_prob_rougeL_conf, norm_prob_rougeL_dens, marker='v', label = 'norm_prob')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.title(image_title)
plt.legend(loc=2)
plt.show()
plt.savefig(save_file+'.png')
plt.close()


# Sample
ppl_exact_conf, ppl_exact_acc, ppl_exact_dens = ece_curve(np.array(sample_ppl_prob), np.array(sample_exact_match))
prob_exact_conf, prob_exact_acc, prob_exact_dens = ece_curve(np.array(sample_prob), np.array(sample_exact_match))
norm_ppl_exact_conf, norm_ppl_exact_acc, norm_ppl_exact_dens = ece_curve(np.array(sample_norm_ppl_prob), np.array(sample_exact_match))
norm_prob_exact_conf, norm_prob_exact_acc, norm_prob_exact_dens = ece_curve(np.array(sample_norm_prob), np.array(sample_exact_match))
ppl_rougeL_conf, ppl_rougeL_acc, ppl_rougeL_dens = ece_curve(np.array(sample_ppl_prob), np.array(sample_rougeL))
prob_rougeL_conf, prob_rougeL_acc, prob_rougeL_dens = ece_curve(np.array(sample_prob), np.array(sample_rougeL))
norm_ppl_rougeL_conf, norm_ppl_rougeL_acc, norm_ppl_rougeL_dens = ece_curve(np.array(sample_norm_ppl_prob), np.array(sample_rougeL))
norm_prob_rougeL_conf, norm_prob_rougeL_acc, norm_prob_rougeL_dens = ece_curve(np.array(sample_norm_prob), np.array(sample_rougeL))

image_title = 'SAMPLE_EXACT ECE'
save_file = os.path.join(save_path, image_title)
plt.plot(ppl_exact_conf, ppl_exact_acc, marker='o', label = 'ppl')
plt.plot(prob_exact_conf, prob_exact_acc, marker='.', label = 'prob')
plt.plot(norm_ppl_exact_conf, norm_ppl_exact_acc, marker=',', label = 'norm_ppl')
plt.plot(norm_prob_exact_conf, norm_prob_exact_acc, marker='v', label = 'norm_prob')
plt.plot([0, 1], [0, 1])
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title(image_title)
plt.legend(loc=2)
plt.show()
plt.savefig(save_file+'.png')
plt.close()

image_title = 'SAMPLE_ROUGE ECE'
save_file = os.path.join(save_path, image_title)
plt.plot(ppl_rougeL_conf, ppl_rougeL_acc, marker='o', label = 'ppl')
plt.plot(prob_rougeL_conf, prob_rougeL_acc, marker='.', label = 'prob')
plt.plot(norm_ppl_rougeL_conf, norm_ppl_rougeL_acc, marker=',', label = 'norm_ppl')
plt.plot(norm_prob_rougeL_conf, norm_prob_rougeL_acc, marker='v', label = 'norm_prob')
plt.plot([0, 1], [0, 1])
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title(image_title)
plt.legend(loc=2)
plt.show()
plt.savefig(save_file+'.png')
plt.close()

image_title = 'SAMPLE Density'
save_file = os.path.join(save_path, image_title)
plt.plot(ppl_rougeL_conf, ppl_rougeL_dens, marker='o', label = 'ppl')
plt.plot(prob_rougeL_conf, prob_rougeL_dens, marker='.', label = 'prob')
plt.plot(norm_ppl_rougeL_conf, norm_ppl_rougeL_dens, marker=',', label = 'norm_ppl')
plt.plot(norm_prob_rougeL_conf, norm_prob_rougeL_dens, marker='v', label = 'norm_prob')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.title(image_title)
plt.legend(loc=2)
plt.show()
plt.savefig(save_file+'.png')
plt.close()