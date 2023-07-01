
def add_args(parser):
    # General setting
    parser.add_argument('--save_dir', type=str, default='results', help='results directory')
    parser.add_argument('--hf_dataset_cache', type=str, default='resluts/hf_dataset_cache', help='cache directory')
    parser.add_argument('--hf_cache', type=str, default='resluts/hf_cache', help='cache directory')
    parser.add_argument('--gpu', type=int, default=7, help='gpu device id')
    parser.add_argument('--log_interval', type=int, default=5, help='print interval')
    parser.add_argument('--run_name', type=str, default='run_1')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    # Dataset
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='data directory')
    parser.add_argument('--dataset', type=str, default='coqa', help='raw dataset:[coqa, trivia_qa, hotpotqa, nq, narrativeqa, ms_marco]')
    parser.add_argument('--fraction_of_train_data', type=float, default=1)
    parser.add_argument('--reparse_dataset', type=bool, default=False)
    parser.add_argument('--clean_dataset', type=bool, default=True)
    # Model
    parser.add_argument('--semantic_model', type=str, default='deberta', help='the model used to judge semantic entailment: deberta')
    parser.add_argument('--bertscore_model', type=str, default='bert', help='the model used to calculate bertscore: [bert]')
    parser.add_argument('--generate_model', type=str, default='opt-2.7b', help='the model used in natural language generation tasks: [opt-1.3b, opt-2.7b, opt-6.7b, opt-13b, opt-30b, gpt-3]')
    parser.add_argument('--num_generations_per_prompt', type=int, default=5, help='num of generation answers for one prompt')
    parser.add_argument('--one_time_generate_num', type=int, default=2, help='num of the most likely generations in one time, no more than num_beams')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--top_p', type=float, default=1.0)
    # Generation
    parser.add_argument('--type_of_question', type=str)
    parser.add_argument('--generate_without_metric', type=bool, default=True)
    # Calculate
    parser.add_argument('--calculate_ordinary', type=bool, default = False)
    parser.add_argument('--calculate_semantic', type=bool, default = True)


device_map = {
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.layers.0': 0,
    'model.decoder.layers.1': 0,
    'model.decoder.layers.2': 0,
    'model.decoder.layers.3': 0,
    'model.decoder.layers.4': 0,
    'model.decoder.layers.5': 0,
    'model.decoder.layers.6': 0,
    'model.decoder.layers.7': 0,
    'model.decoder.layers.8': 0,
    'model.decoder.layers.9': 0,
    'model.decoder.layers.10': 0,
    'model.decoder.layers.11': 0,
    'model.decoder.layers.12': 0,
    'model.decoder.layers.13': 0,
    'model.decoder.layers.14': 0,
    'model.decoder.layers.15': 0,
    'model.decoder.layers.16': 0,
    'model.decoder.layers.17': 0,
    'model.decoder.layers.18': 0,
    'model.decoder.layers.19': 0,
    'model.decoder.layers.20': 0,
    'model.decoder.layers.21': 0,
    'model.decoder.layers.22': 0,
    'model.decoder.layers.23': 0,
    'model.decoder.layers.24': 0,
    'model.decoder.layers.25': 1,
    'model.decoder.layers.26': 1,
    'model.decoder.layers.27': 1,
    'model.decoder.layers.28': 1,
    'model.decoder.layers.29': 1,
    'model.decoder.layers.30': 1,
    'model.decoder.layers.31': 1,
    'model.decoder.layers.32': 1,
    'model.decoder.layers.33': 1,
    'model.decoder.layers.34': 1,
    'model.decoder.layers.35': 1,
    'model.decoder.layers.36': 1,
    'model.decoder.layers.37': 1,
    'model.decoder.layers.38': 1,
    'model.decoder.layers.39': 1,
    'model.decoder.layers.40': 1,
    'model.decoder.layers.41': 1,
    'model.decoder.layers.42': 1,
    'model.decoder.layers.43': 1,
    'model.decoder.layers.44': 1,
    'model.decoder.layers.45': 1,
    'model.decoder.layers.46': 1,
    'model.decoder.layers.47': 1,
    'model.decoder.layers.48': 1,
    'model.decoder.final_layer_norm': 1,
    'lm_head': 1
}

