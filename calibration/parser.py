
def add_args(parser):
    # General setting
    parser.add_argument('--save_dir', type=str, default='resluts', help='results directory')
    parser.add_argument('--hf_dataset_cache', type=str, default='resluts/hf_dataset_cache', help='cache directory')
    parser.add_argument('--hf_cache', type=str, default='resluts/hf_cache', help='cache directory')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    # Dataset
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--dataset', type=str, default='coqa', help='raw dataset:[coqa, trivia_qa, hotpotqa, nq, narrativeqa, ms_marco]')
    # Model
    parser.add_argument('--semantic_model', type=str, default='deberta', help='the model used to judge semantic entailment: deberta')
    parser.add_argument('--bertscore_model', type=str, default='bert', help='the model used to calculate bertscore: [bert]')
    parser.add_argument('--generate_model', type=str, default='opt-2.7b', help='the model used in natural language generation tasks: [opt2.7b, opt6.7b, opt13b, opt30b, gpt3]')
    parser.add_argument('--run_id', type=str, default='run_1')