"""
    1. parse_dataset:
        parse raw QA dataset:
            story: 
            question: 
            answer: direct answer
            all_answers: 
            exact_match: is all answers exact match
            rouge1: rouge1 of all answers
            rouge2: rouge2 of all answers
            rougeL: rougeL of all answers
            bertscore: bertscore of all answers
            semantic_entail_variability: is all answers the same semantic entailment variability
            semantic_uncontradict_variability: is all answers the same semantic uncontradictory variability
            id: question id

    2. clean_data:
        clean data, including delete minor parts in sequences and combine duplicated sequences
"""

import json
import argparse
import evaluate
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bert_score import score
import os

from parser import add_args


def main():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Experiments')
    add_args(parser)
    global args, device
    args = parser.parse_args()
    print(args)
    # Device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(int(args.gpu))
    # Save Path
    save_path = os.path.join(args.save_dir, args.dataset)
    # Dataset
    if args.dataset=='coqa':
        with open(f'{args.data_dir}/coqa-dev-v1.0.json', 'r') as infile:
            data = json.load(infile)['data']
    # Model
    if args.semantic_model == 'deberta':
        semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to('cuda:0')

    if args.bertscore_model == 'bert':
        bert_model_name = 'bert-base-cased'
    # Parse dataset
    dataset = parse_dataset(data, semantic_tokenizer, semantic_model, bert_model_name)
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(dataset_df)
    dataset.save_to_disk(save_path)



def parse_dataset(data, semantic_tokenizer, semantic_model, bert_model_name):
    # Metric
    rouge = evaluate.load('rouge')

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
    dataset['semantic_unentail_cnt'] = []
    dataset['semantic_contradict_cnt'] = []
    dataset['id'] = []

    for sample_id, sample in enumerate(data):
        story = sample['story']
        questions = sample['questions']
        answer = sample['answers']
        additional_answers = sample['additional_answers']

        for question_index, question in enumerate(questions):
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
                semantic_unentail_cnt = prediction[0] - torch.argmax(prediction, dim=1).eq(2).float().sum().item()
                dataset['semantic_unentail_cnt'].append(semantic_unentail_cnt)
                dataset['semantic_contradict_cnt'].append(semantic_contradict_cnt)
                results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
                dataset['rouge1'].append(results['rouge1'])
                dataset['rouge2'].append(results['rouge2'])
                dataset['rougeL'].append(results['rougeL'])
            else:
                dataset['exact_match'].append(True)
                dataset['rouge1'].append(1)
                dataset['rouge2'].append(1)
                dataset['rougeL'].append(1)
                dataset['bertscore'].append(1)
                dataset['semantic_unentail_cnt'].append(0)
                dataset['semantic_contradict_cnt'].append(0)

    return dataset



if __name__ == '__main__':
    main()
