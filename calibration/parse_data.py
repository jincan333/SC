"""
    1. parse_dataset:
        parse raw QA dataset:
            story: 
            question: 
            answer: direct answer
            all_answers: 
            exact_match: exact match score
            rouge1: rouge1 of all answers
            rouge2: rouge2 of all answers
            rougeL: rougeL of all answers
            bertscore: bertscore of all answers
            semantic_entail_variability: is all answers the same semantic entailment variability
            semantic_uncontradict_variability: is all answers the same semantic uncontradictory variability
            id: question id

    2. clean_data:
        clean data, including delete minor parts in sequences and combine duplicated sequences, blankspace etc
"""

import json
import argparse
import evaluate
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import os
import logging
import time
import accelerate
import wandb
import pickle

logging.getLogger('transformers').setLevel(logging.ERROR)
from parser_config import add_args, device_map
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Experiments')
    add_args(parser)
    global args, device
    args = parser.parse_args()
    print(args)
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(int(args.gpu))

    wandb.init(project='nlg_uncertainty', id=args.run_name, config=args, resume='allow')
    run_name = wandb.run.name
    os.environ["HF_DATASETS_CACHE"] = args.hf_dataset_cache
    
    # Random seed
    set_seed(args.seed)
    # Dataset
    if args.dataset=='coqa':
        with open(f'{args.dataset_dir}/coqa-dev-v1.0.json', 'r') as infile:
            data = json.load(infile)['data']
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
            

    # Parse dataset
    if args.reparse_dataset:
        # Save Path
        save_path = os.path.join(args.save_dir, args.dataset, 'parse')
        dataset = parse_dataset(data, semantic_tokenizer, semantic_model, bert_model_name, device, args)
        dataset_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(dataset_df)
        dataset.save_to_disk(save_path)
        print('save file:', save_path)

    # Clean data
    if args.clean_dataset:
        # Save Path
        save_path = os.path.join(args.save_dir, args.dataset, args.generate_model, args.run_name, str(args.one_time_generate_num)+'_'+str(args.num_generations_per_prompt), 'generation')
        with open(save_path+'/generations_without_metric.pkl', 'rb') as infile:
            sequences = pickle.load(infile)
        cleaned_sequences = clean_data(generate_tokenizer, sequences, save_path)
        with open(save_path+'/cleaned_generations_without_metric.pkl', 'wb') as outfile:
            pickle.dump(cleaned_sequences, outfile)
        print('save file:', save_path+'/cleaned_generations_without_metric.pkl')

def clean_data(generate_tokenizer, sequences, save_path):
    print('start clean data:', save_path)
    cleaned_sequences = []
    q_cnt = 0
    start = time.time()
    for sample in sequences:
        
        question = sample['question']
        most_likely_generations_texts = sample['most_likely_generations_texts']
        cleaned_most_likely_generations = torch.ones_like(sample['most_likely_generations'])
        cleaned_most_likely_generations_texts = []
        max_len_of_most_likely_generations = cleaned_most_likely_generations.shape[-1]
        generated_texts = sample['generated_texts']
        cleaned_generations = torch.ones_like(sample['generations'])
        cleaned_generated_texts = []
        max_len_of_generations = cleaned_generations.shape[-1]
        strings_to_filter_on = ['.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:','ANSWER:']
        # Clean most likelt generation texts
        for i, generated_text in enumerate(most_likely_generations_texts):
            generated_text = generated_text.strip()
            for string in strings_to_filter_on:
                if string in generated_text:
                    generated_text = generated_text.split(string)[0]
                    generated_text = generated_text.strip()
            cleaned_most_likely_generations_texts.append(generated_text)
            clean_most_likely_generations_ids = torch.cat([sample['prompt'].to(device), torch.tensor(generate_tokenizer(generated_text)['input_ids'][1:], device=device)])
            cleaned_most_likely_generations[i, :min(len(clean_most_likely_generations_ids), max_len_of_most_likely_generations)] = clean_most_likely_generations_ids[:max_len_of_most_likely_generations]
        # Clean generated texts
        for i, generated_text in enumerate(generated_texts):
            generated_text = generated_text.strip()
            for string in strings_to_filter_on:
                if string in generated_text:
                    generated_text = generated_text.split(string)[0]
                    generated_text = generated_text.strip()
            cleaned_generated_texts.append(generated_text)
            clean_ids = torch.cat([sample['prompt'].to(device), torch.tensor(generate_tokenizer(generated_text)['input_ids'][1:], device=device)])
            cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]
        sample['cleaned_most_likely_generations_texts'] = cleaned_most_likely_generations_texts
        sample['cleaned_most_likely_generations'] = cleaned_most_likely_generations
        sample['cleaned_generated_texts'] = cleaned_generated_texts
        sample['cleaned_generations'] = cleaned_generations
        cleaned_sequences.append(sample)
        q_cnt += 1
        if (q_cnt % (args.log_interval*10)) == 0 or q_cnt== 1:
            end = time.time()
            print(
                f'current question: [{q_cnt}/{len(sequences)}]\n'
                f'Time {end-start:.2f}\n'
            )
            start = time.time()
    
    return cleaned_sequences



def parse_dataset(data, semantic_tokenizer, semantic_model, bert_model_name, device, args):
    print('Start parse dataset:   ', args.dataset)
    start = time.time()
    # Metric
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load('exact_match')

    # Initialize
    dataset = {}
    dataset['story'] = []
    dataset['question'] = []
    dataset['answer'] = []
    dataset['all_answers'] = []
    dataset['exact_match'] = []
    dataset['rouge1'] = []
    dataset['rouge2'] = []
    dataset['rougeL'] = []
    dataset['bertscore'] = []
    dataset['semantic_unentail_rate'] = []
    dataset['semantic_contradict_rate'] = []
    dataset['id'] = []

    q_cnt = 0
    exact_match_cnt = 0
    rouge1_sum = 0
    rouge2_sum = 0
    rougeL_sum = 0
    bertscore_sum = 0
    semantic_unentail_rate_sum = 0
    semantic_contradict_rate_sum = 0
    for sample_id, sample in enumerate(data):
        story = sample['story']
        questions = sample['questions']
        answer = sample['answers']
        additional_answers = sample['additional_answers']

        for question_index, question in enumerate(questions):
            q_cnt += 1
            dataset['story'].append(story)
            dataset['question'].append(question['input_text'])
            dataset['answer'].append({
                'text': answer[question_index]['input_text'],
                'answer_start': answer[question_index]['span_start']
            })
            dataset['id'].append(sample['id'] + '_' + str(question_index))
            
            all_answers_list = [answer[question_index]['input_text']]
            for i in range(3):
                all_answers_list.append(additional_answers[str(i)][question_index]['input_text'])
            all_answers_list = list(set(all_answers_list))
            dataset['all_answers'].append(all_answers_list)
            story = story + ' Q: ' + question['input_text'] + ' A: ' + answer[question_index]['input_text']
            if not story[-1] == '.':
                story = story + '.'

            # This computes the syntactic similarity across the reference answers
            if len(all_answers_list) > 1:
                answer_list_1 = []
                answer_list_2 = []
                inputs = []
                for i, reference_answer in enumerate(all_answers_list):
                    for j in range(len(all_answers_list)):
                        if i != j:
                            answer_list_1.append(all_answers_list[i])
                            answer_list_2.append(all_answers_list[j])
                            qa_1 = question['input_text'] + ' ' + all_answers_list[i]
                            qa_2 = question['input_text'] + ' ' + all_answers_list[j]
                            input = qa_1 + ' [SEP] ' + qa_2
                            inputs.append(input)
                encoded_input = semantic_tokenizer.batch_encode_plus(inputs, padding=True)
                prediction = semantic_model(torch.tensor(encoded_input['input_ids'], device=device))['logits']
                # 0: contradictory, 1: neutral, 2: entailment
                semantic_contradict_cnt = torch.argmax(prediction, dim=1).eq(0).float().sum().item()
                semantic_unentail_cnt = prediction.shape[0] - torch.argmax(prediction, dim=1).eq(2).float().sum().item()
                dataset['exact_match'].append(0)
                results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
                rouge1 = results['rouge1'].mid.fmeasure
                rouge2 = results['rouge2'].mid.fmeasure
                rougeL = results['rougeL'].mid.fmeasure
                # bertscore = score(answer_list_1, answer_list_2, lang='en', model_type = bert_model_name, device = device)[2].mean().item()
                bertscore = 0
                semantic_unentail_rate = semantic_unentail_cnt / len(inputs)
                semantic_contradict_rate = semantic_contradict_cnt / len(inputs)
                dataset['rouge1'].append(rouge1)
                dataset['rouge2'].append(rouge2)
                dataset['rougeL'].append(rougeL)
                dataset['bertscore'].append(bertscore)
                dataset['semantic_unentail_rate'].append(semantic_unentail_rate)
                dataset['semantic_contradict_rate'].append(semantic_contradict_rate)
                rouge1_sum += rouge1
                rouge2_sum += rouge2
                rougeL_sum += rougeL
                bertscore_sum += bertscore
                semantic_unentail_rate_sum += semantic_unentail_rate
                semantic_contradict_rate_sum += semantic_contradict_rate
            else:
                dataset['exact_match'].append(1)
                dataset['rouge1'].append(1)
                dataset['rouge2'].append(1)
                dataset['rougeL'].append(1)
                dataset['bertscore'].append(1)
                dataset['semantic_unentail_rate'].append(0)
                dataset['semantic_contradict_rate'].append(0)
                exact_match_cnt += 1
                rouge1_sum += 1
                rouge2_sum += 1
                rougeL_sum += 1
                bertscore_sum += 1
        if (sample_id) % args.log_interval == 1:
            end = time.time()
            print(
                f'current sample: [{sample_id}/{len(data)}]\n'
                f'question {q_cnt}\n'
                f'exact_match {exact_match_cnt / q_cnt:.4f}\n'
                f'rouge1 {rouge1_sum / q_cnt:.4f}\n'
                f'rouge2 {rouge2_sum / q_cnt:.4f}\n'
                f'rougeL {rougeL_sum / q_cnt:.4f}\n'
                f'bertscore {bertscore_sum / q_cnt:.4f}\n'
                f'semantic_unentail {semantic_unentail_rate_sum / q_cnt:.4f}\n'
                f'semantic_contradict {semantic_contradict_rate_sum / q_cnt:.4f}\n'
                f'Time {end-start:.2f}\n'
            )
            start = time.time()

    return dataset




if __name__ == '__main__':
    main()
