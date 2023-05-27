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

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-2.7b')
parser.add_argument('--generation_model', type=str, default='opt-2.7b')
parser.add_argument('--run_id', type=str, default='run_1')
args = parser.parse_args()
device = 'cuda'
import config
wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')
run_name = wandb.run.name
with open(f'{config.output_dir}/clean/{run_name}/{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl',
          'rb') as infile:
    sequences = pickle.load(infile)
with open(f'{config.output_dir}/clean/{run_name}/{args.generation_model}_generations.pkl', 'rb') as infile:
    generations = pickle.load(infile)
most_confidences = []
second_confidences = []
avg_confidences = []
r03_accuracies = []
r05_accuracies = []
exact_accuracies = []
for i, s in enumerate(sequences):
    g = generations[i]
    most_confidence = s['average_probability_of_most_likely_gen'].cpu().numpy()
    second_confidence = s['average_probability_of_second_most_likely_gen'].cpu().numpy()
    avg_confidence = s['average_probabilities'].cpu().numpy()
    r03_accuracy = 1 if g['rougeL_to_target'] > 0.3 else 0
    r05_accuracy = 1 if g['rougeL_to_target'] > 0.5 else 0
    exact_accuracy = g['exact_match']
    most_confidences.append(most_confidence)
    r03_accuracies.append(r03_accuracy)
    r05_accuracies.append(r05_accuracy)
    exact_accuracies.append(exact_accuracy)



def ece_curve(predictions, true_labels, n_bins=10):
    bin_edges = np.arange(0., 1.1, 1. / n_bins)
    
    accuracies = []
    confidences = []

    for k in range(n_bins):
        idx = np.where((predictions >= bin_edges[k]) & (predictions < bin_edges[k + 1]))[0]
        if len(idx) > 0:
            accuracy = accuracy_score(true_labels[idx], np.round(predictions[idx]))
            confidence = np.mean(predictions[idx])
            accuracies.append(accuracy)
            confidences.append(confidence)

    return confidences, accuracies

# Simulated sample data
predictions = np.array(most_confidences)
r03_true_labels = np.array(r03_accuracies)
r05_true_labels = np.array(r05_accuracies)
exact_true_labels = np.array(exact_accuracies)

# Calculate expected calibration error curve
i = 0
for true_label in [r03_true_labels, r05_true_labels, exact_true_labels]:
    confidences, accuracies = ece_curve(predictions, true_label)
    i+=1
    # Plot the ECE curve
    plt.plot(confidences, accuracies, marker='o')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Expected Calibration Error (ECE) Curve')
    plt.show()
    plt.savefig(f'figures/test_accuracy{i}.png')
    plt.close()