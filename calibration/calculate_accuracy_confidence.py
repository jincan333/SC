import argparse
import csv
import os
import pickle
import random
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import wandb
import logging
import accelerate
import json
import time

logging.getLogger('transformers').setLevel(logging.ERROR)
from parser_config import add_args, device_map
from utils import set_seed

def main():
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
    # Model
    if args.semantic_model == 'deberta':
        semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()
    if args.bertscore_model == 'bert':
        bert_model_name = 'bert-base-uncased'
    if args.generate_model == 'opt-2.7b':
        generate_model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.generate_model}", torch_dtype=torch.float16, cache_dir=args.hf_cache).cuda()
        if args.generate_model == 'opt-30b':
            accelerate.dispatch_model(generate_model, device_map=device_map)
        generate_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.generate_model}", use_fast=False, cache_dir=args.hf_cache)


    # Ordinary accuracy and probability
    if args.calculate_ordinary:
        input_file_pth = os.path.join(save_path,'cleaned_generations_without_metric.pkl')
        with open(input_file_pth, 'rb') as infile:
            sequences = pickle.load(infile)
            print('start ordinary calculate using file ', input_file_pth)
            ordinary_acc_conf = calculate_ordinary(sequences)

        output_file_pth = os.path.join(save_path,'generations_ordinary_accuracy_confidence.pkl')
        with open(output_file_pth, 'wb') as outfile:
            pickle.dump(ordinary_acc_conf, outfile)
            print('ordinary output save path: ', output_file_pth)


    # Semantic cluster, calculate semantic accuracy and probability
    if args.calculate_semantic:
        input_file_pth = os.path.join(save_path,'cleaned_generations_without_metric.pkl')
        with open(input_file_pth, 'rb') as infile:
            sequences = pickle.load(infile)
            print()
            semantic_predictions, result_dict = calculate_semantic(sequences)

        with open(os.path.join(save_path,'semantic_predictions.csv'), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['qa_1', 'qa_2', 'prediction'])
            writer.writerows(semantic_predictions)

        print(result_dict)

        with open(os.path.join(save_path,'generations_similarities.pkl'), 'wb') as outfile:
            pickle.dump(result_dict, outfile)



def calculate_semantic(sequences):

    result_dict = {}
    deberta_predictions = []
    for sample in tqdm(sequences):
        question = sample['question']
        if 'cleaned_generated_texts' in sample:
            generated_texts = sample['cleaned_generated_texts']
        else:
            generated_texts = sample['generated_texts']

        id_ = sample['id'][0]
        unique_generated_texts = list(set(generated_texts))
        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []
        syntactic_similarities = {}
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        for rouge_type in rouge_types:
            syntactic_similarities[rouge_type] = 0.0

        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index

        print('Number of unique answers:', len(unique_generated_texts))
        if len(unique_generated_texts) > 1:
            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):
                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])
                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]
                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = semantic_tokenizer.encode(input, padding=True)
                    prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=device))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)
                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=device))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                    deberta_prediction = 1
                    print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0
                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                    deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])
            rouge = evaluate.load('rouge')
            # Evalauate syntactic similarity
            answer_list_1 = []
            answer_list_2 = []
            for i in generated_texts:
                for j in generated_texts:
                    if i != j:
                        answer_list_1.append(i)
                        answer_list_2.append(j)
            results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
            for rouge_type in rouge_types:
                # syntactic_similarities[rouge_type] = results[rouge_type].mid.fmeasure
                syntactic_similarities[rouge_type] = results[rouge_type]
        result_dict[id_] = {
            'syntactic_similarities': syntactic_similarities,
            'has_semantically_different_answers': has_semantically_different_answers
        }
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
        result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids
    
    return deberta_predictions, result_dict




def calculate_ordinary(sequences):
    start = time.time()
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    q_cnt = 0
    print_ppl_prob = 0
    print_prob = 0
    print_norm_ppl_prob = 0
    pront_norm_prob = 0
    print_exact_match = 0
    print_rouge1 = 0
    print_rouge2 = 0
    print_rougeL = 0
    print_bertscore = 0
    with torch.no_grad():
        result = []
        for sample in sequences:
            result_dict = {}

            id_ = sample['id']
            prompt = sample['prompt']
            question = sample['question']
            all_answers = sample['all_answers']
            most_likely_generations = sample['cleaned_most_likely_generations'].to(device)
            cleaned_most_likely_generations_texts = sample['cleaned_most_likely_generations_texts']
            generations = sample['cleaned_generations'].to(device)
            cleaned_generated_texts = sample['cleaned_generated_texts']
            most_ppl_probabilities = torch.zeros((most_likely_generations.shape[0],))
            most_probabilities = torch.zeros((most_likely_generations.shape[0],))
            most_ppl_unconditioned_probabilities = torch.zeros((most_likely_generations.shape[0],))
            most_unconditioned_probabilities = torch.zeros((most_likely_generations.shape[0],))
            most_norm_ppl_probabilities = torch.zeros((most_likely_generations.shape[0],))
            most_norm_probabilities = torch.zeros((most_likely_generations.shape[0],))

            sample_ppl_probabilities = torch.zeros((generations.shape[0],))
            sample_probabilities = torch.zeros((generations.shape[0],))
            sample_ppl_unconditioned_probabilities = torch.zeros((generations.shape[0],))
            sample_unconditioned_probabilities = torch.zeros((generations.shape[0],))
            sample_norm_ppl_probabilities = torch.zeros((generations.shape[0],))
            sample_norm_probabilities = torch.zeros((generations.shape[0],))
            # calculate most direct prob and ppl prob
            for generation_index in range(most_likely_generations.shape[0]):
                prompt = prompt[prompt != generate_tokenizer.pad_token_id]
                generation = most_likely_generations[generation_index][most_likely_generations[generation_index] != generate_tokenizer.pad_token_id]
                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                target_ids = generation.clone()
                target_ids[:len(prompt)] = -100
                generation_only = generation.clone()[(len(prompt) - 1):]
                model_output = generate_model(torch.reshape(generation, (1, -1)), labels=target_ids)
                unconditioned_model_output = generate_model(torch.reshape(generation_only, (1, -1)), labels=generation_only)
                ppl_probability = torch.exp(-model_output['loss'])
                probability = torch.exp(-model_output['loss'] * (len(generation) - len(prompt))) 
                ppl_unconditioned_probability = torch.exp(-unconditioned_model_output['loss'])
                unconditioned_probability = torch.exp(-unconditioned_model_output['loss'] * (len(generation) - len(prompt))) 

                most_ppl_probabilities[generation_index] = ppl_probability
                most_probabilities[generation_index] = probability
                most_ppl_unconditioned_probabilities[generation_index] = ppl_unconditioned_probability
                most_unconditioned_probabilities[generation_index] = unconditioned_probability
            most_norm_ppl_probabilities = most_ppl_probabilities / most_ppl_probabilities.sum()
            most_norm_probabilities = most_probabilities / most_probabilities.sum()
            # calculate most accuracy
            most_ppl_prob, most_index = torch.max(most_ppl_probabilities, dim=0)
            most_index = most_index.item()
            most_ppl_prob = most_ppl_prob.item()
            most_prob = most_probabilities[most_index].item()
            most_norm_ppl_prob = most_norm_ppl_probabilities[most_index].item()
            most_norm_prob = most_norm_probabilities[most_index].item()
            most_text = cleaned_most_likely_generations_texts[most_index]
            most_exact_match = 0
            most_rouge1 = 0
            most_rouge2 = 0
            most_rougeL = 0
            most_bertscore = 0
            for ans in all_answers:
                preds = [most_text]
                anss = [ans]
                most_exact_match = max(exact_match_metric.compute(predictions=preds, references=anss, ignore_case=True, ignore_punctuation=True)['exact_match'], most_exact_match)
                rouge_results = rouge.compute(predictions=preds, references=anss)
                most_rouge1 = max(most_rouge1, rouge_results['rouge1'].mid.fmeasure)
                most_rouge2 = max(most_rouge2, rouge_results['rouge2'].mid.fmeasure)
                most_rougeL = max(most_rougeL, rouge_results['rougeL'].mid.fmeasure)
                # bertscore = max(score(preds, anss, lang='en', model_type = bert_model_name, device = device)[2].mean().item(), bertscore)

            # calculate sample direct prob and ppl prob
            for generation_index in range(generations.shape[0]):
                prompt = prompt[prompt != generate_tokenizer.pad_token_id]
                generation = generations[generation_index][generations[generation_index] != generate_tokenizer.pad_token_id]

                target_ids = generation.clone()
                target_ids[:len(prompt)] = -100
                generation_only = generation.clone()[(len(prompt) - 1):]
                model_output = generate_model(torch.reshape(generation, (1, -1)), labels=target_ids)
                unconditioned_model_output = generate_model(torch.reshape(generation_only, (1, -1)), labels=generation_only)
                ppl_probability = torch.exp(-model_output['loss'])
                probability = torch.exp(-model_output['loss'] * (len(generation) - len(prompt))) 
                ppl_unconditioned_probability = torch.exp(-unconditioned_model_output['loss'])
                unconditioned_probability = torch.exp(-unconditioned_model_output['loss']  * (len(generation) - len(prompt)))

                sample_ppl_probabilities[generation_index] = ppl_probability
                sample_probabilities[generation_index] = probability
                sample_ppl_unconditioned_probabilities[generation_index] = ppl_unconditioned_probability
                sample_unconditioned_probabilities[generation_index] = unconditioned_probability
            sample_norm_ppl_probabilities = sample_ppl_probabilities / sample_ppl_probabilities.sum()
            sample_norm_probabilities = sample_probabilities / sample_probabilities.sum()
            # calculate sample accuracy
            sample_ppl_prob, sample_index = torch.max(sample_ppl_probabilities, dim=0)
            sample_index = sample_index.item()
            sample_ppl_prob = sample_ppl_prob.item()
            sample_prob = sample_probabilities[sample_index].item()
            sample_norm_ppl_prob = sample_norm_ppl_probabilities[sample_index].item()
            sample_norm_prob = sample_norm_probabilities[sample_index].item()
            sample_text = cleaned_generated_texts[sample_index]
            sample_exact_match = 0
            sample_rouge1 = 0
            sample_rouge2 = 0
            sample_rougeL = 0
            sample_bertscore = 0
            for ans in all_answers:
                preds = [sample_text]
                anss = [ans]
                sample_exact_match = max(exact_match_metric.compute(predictions=preds, references=anss, ignore_case=True, ignore_punctuation=True)['exact_match'], sample_exact_match)
                rouge_results = rouge.compute(predictions=preds, references=anss)
                sample_rouge1 = max(sample_rouge1, rouge_results['rouge1'].mid.fmeasure)
                sample_rouge2 = max(sample_rouge2, rouge_results['rouge2'].mid.fmeasure)
                sample_rougeL = max(sample_rougeL, rouge_results['rougeL'].mid.fmeasure)

            # record
            most_likely_generations = sample['cleaned_most_likely_generations'].to(device)
            cleaned_most_likely_generations_texts = sample['cleaned_most_likely_generations_texts']
            generations = sample['cleaned_generations'].to(device)
            cleaned_generated_texts = sample['cleaned_generated_texts']

            result_dict['id'] = id_
            result_dict['prompt'] = prompt
            result_dict['question'] = question
            result_dict['all_answers'] = all_answers
            result_dict['cleaned_most_likely_generations_texts'] = cleaned_most_likely_generations_texts
            result_dict['cleaned_generated_texts'] = cleaned_generated_texts
            result_dict['most_likely_generations'] = most_likely_generations
            result_dict['generations'] = generations

            result_dict['most_ppl_probabilities'] = most_ppl_probabilities
            result_dict['most_probabilities'] = most_probabilities
            result_dict['most_norm_ppl_probabilities'] = most_norm_ppl_probabilities
            result_dict['most_norm_probabilities'] = most_norm_probabilities
            result_dict['most_ppl_unconditioned_probabilities'] = most_ppl_unconditioned_probabilities
            result_dict['most_unconditioned_probabilities'] = most_unconditioned_probabilities
            result_dict['most_index'] = most_index
            result_dict['most_text'] = most_text
            result_dict['most_ppl_prob'] = most_ppl_prob
            result_dict['most_prob'] = most_prob
            result_dict['most_norm_ppl_prob'] = most_norm_ppl_prob
            result_dict['most_norm_prob'] = most_norm_prob
            result_dict['most_exact_match'] = most_exact_match
            result_dict['most_rouge1'] = most_rouge1
            result_dict['most_rouge2'] = most_rouge2
            result_dict['most_rougeL'] = most_rougeL
            result_dict['most_bertscore'] = most_bertscore

            result_dict['sample_ppl_probabilities'] = sample_ppl_probabilities
            result_dict['sample_probabilities'] = sample_probabilities
            result_dict['sample_norm_ppl_probabilities'] = sample_norm_ppl_probabilities
            result_dict['sample_norm_probabilities'] = sample_norm_probabilities
            result_dict['sample_ppl_unconditioned_probabilities'] = sample_ppl_unconditioned_probabilities
            result_dict['sample_unconditioned_probabilities'] = sample_unconditioned_probabilities
            result_dict['sample_index'] = sample_index
            result_dict['sample_text'] = sample_text
            result_dict['sample_ppl_prob'] = sample_ppl_prob
            result_dict['sample_prob'] = sample_prob
            result_dict['sample_norm_ppl_prob'] = sample_norm_ppl_prob
            result_dict['sample_norm_prob'] = sample_norm_prob
            result_dict['sample_exact_match'] = sample_exact_match
            result_dict['sample_rouge1'] = sample_rouge1
            result_dict['sample_rouge2'] = sample_rouge2
            result_dict['sample_rougeL'] = sample_rougeL
            result_dict['sample_bertscore'] = sample_bertscore
            
            result.append(result_dict)
            q_cnt += 1
            print_ppl_prob += most_ppl_prob
            print_prob += most_prob
            print_norm_ppl_prob += most_norm_ppl_prob
            pront_norm_prob += most_norm_prob
            print_exact_match += most_exact_match
            print_rouge1 += most_rouge1
            print_rouge2 += most_rouge2
            print_rougeL += most_rougeL
            print_bertscore += 0
            if (q_cnt % (args.log_interval*10)) == 0 or q_cnt== 1:
                end = time.time()
                print(
                    f'current question: [{q_cnt}/{len(sequences)}]\n'
                    f'most_ppl_prob {print_ppl_prob / q_cnt:.4f}\n'
                    f'most_prob {print_prob / q_cnt:.4f}\n'
                    f'most_norm_ppl_prob {print_norm_ppl_prob / q_cnt:.4f}\n'
                    f'most_norm_prob {pront_norm_prob / q_cnt:.4f}\n'
                    f'most_exact_match {print_exact_match / q_cnt:.4f}\n'
                    f'most_rouge1 {print_rouge1 / q_cnt:.4f}\n'
                    f'most_rouge2 {print_rouge2 / q_cnt:.4f}\n'
                    f'most_rougeL {print_rougeL / q_cnt:.4f}\n'
                    f'most_bertscore {print_bertscore / q_cnt:.4f}\n'
                    f'Time {end-start:.2f}\n'
                )
                start = time.time()

        return result



if __name__ == '__main__':
    main()