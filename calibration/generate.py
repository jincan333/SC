import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize
import accelerate
import datasets
import evaluate
import numpy as np
import torch
import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
from bert_score import score

logging.getLogger('transformers').setLevel(logging.ERROR)
from parser_config import add_args, device_map
from utils import set_seed

def main():
    global args, device, period_token_id, question_framing_ids, id_to_question_mapping
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
    save_path = os.path.join(args.save_dir, args.dataset, args.generate_model, args.run_name, str(args.one_time_generate_num)+'_'+str(args.num_generations_per_prompt)
                            , 'generation')
    # Random seed
    set_seed(args.seed)
    # Model
    generate_model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.generate_model}", torch_dtype=torch.float16, cache_dir=args.hf_cache).cuda()
    if args.generate_model == 'opt-30b':
        accelerate.dispatch_model(generate_model, device_map=device_map)
    generate_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.generate_model}", use_fast=False, cache_dir=args.hf_cache)
    if args.bertscore_model == 'bert':
        bert_model_name = 'bert-base-uncased'
    # Dataset
    if args.dataset == 'coqa':
        dataset = datasets.load_from_disk(os.path.join(args.save_dir, args.dataset, 'parse'))
        id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
    elif args.dataset == 'trivia_qa':
        dataset = datasets.load_from_disk(os.path.join(args.save_dir, args.dataset, 'parse'))

    if args.fraction_of_train_data < 1.0:
        train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_train_data), seed=args.seed)['train']
    else:
        train_dataset = dataset

    if args.dataset == 'coqa':
        questions = encode_and_format_dataset(train_dataset, generate_tokenizer)
    elif args.dataset == 'trivia_qa':
        questions = train_dataset

    dataloader = torch.utils.data.DataLoader(questions, batch_size=1)
    period_token_id = generate_tokenizer('. ')['input_ids'][1]
    eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[generate_tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]

    if args.generate_without_metric:
        sequences = generate_without_metric(generate_model, generate_tokenizer, dataloader, args)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(save_path+'/generations_without_metric.pkl', 'wb') as outfile:
            pickle.dump(sequences, outfile)
        print('save file: ', save_path+'/generations_without_metric.pkl')
    else:
        sequences = get_generations(generate_model, generate_tokenizer, dataloader, bert_model_name, args)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(save_path+'/generations.pkl', 'wb') as outfile:
            pickle.dump(sequences, outfile)
        print('save file: ', save_path+'/generations.pkl')


def encode_and_format_dataset(dataset, generate_tokenizer):

    def encode(examples):
        return generate_tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)

    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset


def get_generations(model, generate_tokenizer, dataloader, bert_model_name, args):
    # Metric
    squad_metric = evaluate.load("squad")
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    """For a given model, produce a number of generation """
    print('Start generate answers:   ', args.dataset)
    start = time.time()
    with torch.no_grad():
        max_length_of_generated_sequence = 256
        sequences = []
        q_cnt = 0
        exact_match_cnt = 0
        rouge1_sum = 0
        rouge2_sum = 0
        rougeL_sum = 0
        bertscore_sum = 0
        semantic_unentail_rate_sum = 0
        semantic_contradict_rate_sum = 0
        for batch in dataloader:
            q_cnt += 1
            input_ids = torch.cat(batch['input_ids']).to(device).reshape(1, -1) if args.dataset == 'trivia_qa' else batch['input_ids'].to(device)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(
                    input_ids,
                    num_beams=args.num_beams,
                    num_return_sequences=args.one_time_generate_num,
                    do_sample=False,
                    max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                    eos_token_id=period_token_id,
                    bad_words_ids=question_framing_ids,
                )
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(
                    input_ids,
                    num_beams=1,
                    do_sample=False,
                    max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                    eos_token_id=period_token_id,
                    bad_words_ids=question_framing_ids
                )
            
            input_length = input_ids.shape[1] if args.dataset == 'trivia_qa' else batch['input_ids'].shape[1]
            generations = torch.ones((args.num_generations_per_prompt, input_length + max_length_of_generated_sequence), dtype=torch.long, device=device)
            for i in range(args.num_generations_per_prompt):
                try:
                    generation = model.generate(
                        input_ids,
                        do_sample=True,
                        num_return_sequences=1,
                        num_beams=args.num_beams,
                        max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                        eos_token_id=period_token_id,
                        temperature=args.temperature,
                        bad_words_ids=question_framing_ids,
                        top_p=args.top_p
                    )
                    generations[i, :generation.shape[1]] = generation
                except:
                    print('num_generations_per_prompt calculate error, q_cnt: ',q_cnt)
                    try:
                        generation = model.generate(
                            input_ids,
                            do_sample=True,
                            num_return_sequences=1,
                            num_beams=args.num_beams,
                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                            eos_token_id=period_token_id,
                            temperature=1,
                            bad_words_ids=question_framing_ids,
                            top_p=args.top_p
                        )
                        generations[i, :generation.shape[1]] = generation
                    except:
                        generation = model.generate(
                            input_ids,
                            do_sample=True,
                            num_return_sequences=1,
                            num_beams=args.num_beams,
                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                            eos_token_id=period_token_id,
                            temperature=2,
                            bad_words_ids=question_framing_ids,
                            top_p=args.top_p
                        )
                        generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, args.num_generations_per_prompt, generations.shape[-1]))
            for i in range(generations.shape[0]):
                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'most_likely_generations': most_likely_generation.to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                elif args.dataset == 'trivia_qa':
                    few_shot_question = generate_tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': generate_tokenizer.decode(input_ids[0]),
                        'question': question
                    }
                
                # Reference answer info
                sequence_dict['exact_match_ref'] = batch['exact_match']
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_ref'] = batch[rouge_type]
                    else:
                        sequence_dict[rouge_type + '_ref'] = None
                    sequence_dict[rouge_type + '_pred'] = 0.0
                sequence_dict['bertscore_ref'] = batch['bertscore']
                sequence_dict['semantic_unentail_rate_ref'] = batch['semantic_unentail_rate']
                sequence_dict['semantic_contradict_rate_ref'] = batch['semantic_contradict_rate']
                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']
                sequence_dict['all_answers'] = [x[0] for x in batch['all_answers']] if args.dataset == 'coqa' else None
                reference_answers = [x[0] for x in batch['all_answers']] if args.dataset == 'coqa' else batch['answer']
                # Prediction info
                most_likely_generations_texts = []
                for generation in most_likely_generation:
                    most_likely_generations_texts.append(generate_tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))
                sequence_dict['most_likely_generations_texts'] = most_likely_generations_texts

                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = generate_tokenizer.decode( most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)
                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = generate_tokenizer.decode(most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)
                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(generate_tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))
                sequence_dict['generated_texts'] = generated_texts
                # Calculate accuracy
                # Most likely answers
                most_likely_exact_match = []
                most_likely_rouge1 = []
                most_likely_rouge2 = []
                most_likely_rougeL = []
                most_likely_bertscore = []
                for pred in most_likely_generations_texts:
                    exact_match = 0
                    rouge1 = 0
                    rouge2 = 0
                    rougeL = 0
                    bertscore = 0
                    for ans in reference_answers:
                        preds = [pred]
                        anss = [ans]
                        exact_match = max(exact_match_metric.compute(predictions=preds, references=anss, ignore_case=True, ignore_punctuation=True)['exact_match'], exact_match)
                        rouge_results = rouge.compute(predictions=preds, references=anss)
                        rouge1 = max(rouge1, rouge_results['rouge1'].mid.fmeasure)
                        rouge2 = max(rouge2, rouge_results['rouge2'].mid.fmeasure)
                        rougeL = max(rougeL, rouge_results['rougeL'].mid.fmeasure)
                        # bertscore = max(score(preds, anss, lang='en', model_type = bert_model_name, device = device)[2].mean().item(), bertscore)
                        bertscore = 0
                    most_likely_exact_match.append(exact_match)
                    most_likely_rouge1.append(rouge1)
                    most_likely_rouge2.append(rouge2)
                    most_likely_rougeL.append(rougeL)
                    most_likely_bertscore.append(bertscore)
                sequence_dict['most_likely_exact_match'] = most_likely_exact_match
                sequence_dict['most_likely_rouge1'] = most_likely_rouge1
                sequence_dict['most_likely_rouge2'] = most_likely_rouge2
                sequence_dict['most_likely_rougeL'] = most_likely_rougeL
                sequence_dict['most_likely_bertscore'] = most_likely_bertscore

                # Diverse answers
                diverse_exact_match = []
                diverse_rouge1 = []
                diverse_rouge2 = []
                diverse_rougeL = []
                diverse_bertscore = []
                for pred in generated_texts:
                    exact_match = 0
                    rouge1 = 0
                    rouge2 = 0
                    rougeL = 0
                    bertscore = 0
                    for ans in reference_answers:
                        preds = [pred]
                        anss = [ans]
                        exact_match = max(exact_match_metric.compute(predictions=preds, references=anss, ignore_case=True, ignore_punctuation=True)['exact_match'], exact_match)
                        rouge_results = rouge.compute(predictions=preds, references=anss)
                        rouge1 = max(rouge1, rouge_results['rouge1'].mid.fmeasure)
                        rouge2 = max(rouge2, rouge_results['rouge2'].mid.fmeasure)
                        rougeL = max(rougeL, rouge_results['rougeL'].mid.fmeasure)
                        # bertscore = max(score(preds, anss, lang='en', model_type = bert_model_name, device = device)[2].mean().item(), bertscore)
                        bertscore = 0
                    diverse_exact_match.append(exact_match)
                    diverse_rouge1.append(rouge1)
                    diverse_rouge2.append(rouge2)
                    diverse_rougeL.append(rougeL)
                    diverse_bertscore.append(bertscore)
                sequence_dict['diverse_exact_match'] = diverse_exact_match
                sequence_dict['diverse_rouge1'] = diverse_rouge1
                sequence_dict['diverse_rouge2'] = diverse_rouge2
                sequence_dict['diverse_rougeL'] = diverse_rougeL
                sequence_dict['diverse_bertscore'] = diverse_bertscore
                sequences.append(sequence_dict)

                exact_match_cnt += most_likely_exact_match[0]
                rouge1_sum += most_likely_rouge1[0]
                rouge2_sum += most_likely_rouge2[0]
                rougeL_sum += most_likely_rougeL[0]
                bertscore_sum += most_likely_bertscore[0]
                if (q_cnt % (args.log_interval*10)) == 0 or q_cnt== 1:
                    end = time.time()
                    print(
                        f'current question: [{q_cnt}/{len(dataloader)}]\n'
                        f'exact_match {exact_match_cnt / q_cnt:.4f}\n'
                        f'rouge1 {rouge1_sum / q_cnt:.4f}\n'
                        f'rouge2 {rouge2_sum / q_cnt:.4f}\n'
                        f'rougeL {rougeL_sum / q_cnt:.4f}\n'
                        f'bertscore {bertscore_sum / q_cnt:.4f}\n'
                        f'Time {end-start:.2f}\n'
                    )
                    start = time.time()

    return sequences


def generate_without_metric(model, generate_tokenizer, dataloader, args):
    """For a given model, produce a number of generation """
    print('Start generate answers:   ', args.dataset)
    start = time.time()
    with torch.no_grad():
        max_length_of_generated_sequence = 256
        sequences = []
        q_cnt = 0
        for batch in dataloader:
            q_cnt += 1
            input_ids = torch.cat(batch['input_ids']).to(device).reshape(1, -1) if args.dataset == 'trivia_qa' else batch['input_ids'].to(device)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(
                    input_ids,
                    num_beams=args.num_beams,
                    num_return_sequences=args.one_time_generate_num,
                    do_sample=False,
                    max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                    eos_token_id=period_token_id,
                    bad_words_ids=question_framing_ids,
                )
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(
                    input_ids,
                    num_beams=1,
                    do_sample=False,
                    max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                    eos_token_id=period_token_id,
                    bad_words_ids=question_framing_ids
                )
            
            input_length = input_ids.shape[1] if args.dataset == 'trivia_qa' else batch['input_ids'].shape[1]
            generations = torch.ones((args.num_generations_per_prompt, input_length + max_length_of_generated_sequence), dtype=torch.long, device=device)
            for i in range(args.num_generations_per_prompt):
                try:
                    generation = model.generate(
                        input_ids,
                        do_sample=True,
                        num_return_sequences=1,
                        num_beams=args.num_beams,
                        max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                        eos_token_id=period_token_id,
                        temperature=args.temperature,
                        bad_words_ids=question_framing_ids,
                        top_p=args.top_p
                    )
                    generations[i, :generation.shape[1]] = generation
                except:
                    print('num_generations_per_prompt calculate error, q_cnt: ',q_cnt)
                    try:
                        generation = model.generate(
                            input_ids,
                            do_sample=True,
                            num_return_sequences=1,
                            num_beams=args.num_beams,
                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                            eos_token_id=period_token_id,
                            temperature=1,
                            bad_words_ids=question_framing_ids,
                            top_p=args.top_p
                        )
                        generations[i, :generation.shape[1]] = generation
                    except:
                        generation = model.generate(
                            input_ids,
                            do_sample=True,
                            num_return_sequences=1,
                            num_beams=args.num_beams,
                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                            eos_token_id=period_token_id,
                            temperature=2,
                            bad_words_ids=question_framing_ids,
                            top_p=args.top_p
                        )
                        generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, args.num_generations_per_prompt, generations.shape[-1]))
            for i in range(generations.shape[0]):
                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'most_likely_generations': most_likely_generation.to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                elif args.dataset == 'trivia_qa':
                    few_shot_question = generate_tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': generate_tokenizer.decode(input_ids[0]),
                        'question': question
                    }
                
                # Reference answer info
                sequence_dict['exact_match_ref'] = batch['exact_match']
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_ref'] = batch[rouge_type]
                    else:
                        sequence_dict[rouge_type + '_ref'] = None
                    sequence_dict[rouge_type + '_pred'] = 0.0
                sequence_dict['bertscore_ref'] = batch['bertscore']
                sequence_dict['semantic_unentail_rate_ref'] = batch['semantic_unentail_rate']
                sequence_dict['semantic_contradict_rate_ref'] = batch['semantic_contradict_rate']
                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']
                sequence_dict['all_answers'] = [x[0] for x in batch['all_answers']] if args.dataset == 'coqa' else None
                reference_answers = [x[0] for x in batch['all_answers']] if args.dataset == 'coqa' else batch['answer']
                # Prediction info
                most_likely_generations_texts = []
                for generation in most_likely_generation:
                    most_likely_generations_texts.append(generate_tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))
                sequence_dict['most_likely_generations_texts'] = most_likely_generations_texts

                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = generate_tokenizer.decode( most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)
                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = generate_tokenizer.decode(most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)
                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(generate_tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))
                sequence_dict['generated_texts'] = generated_texts
                sequences.append(sequence_dict)

                if (q_cnt % (args.log_interval*10)) == 0 or q_cnt== 1:
                    end = time.time()
                    print(
                        f'current question: [{q_cnt}/{len(dataloader)}]\n'
                        f'Time {end-start:.2f}\n'
                    )
                    start = time.time()

    return sequences

if __name__ == '__main__':
    main()