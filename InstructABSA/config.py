from argparse import ArgumentParser

class Config(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.model_checkpoint = None
        self.inst_type = None
        self.experiment_name = None
        self.task = None
        self.output_dir = None
        self.id_tr_data_path = None
        self.id_te_data_path = None
        self.set_instruction_key = None
        self.ood_tr_data_path = None
        self.ood_te_data_path = None
        self.output_path = None
        self.sample_size = 1
        self.evaluation_strategy = None
        self.learning_rate = 5e-5
        self.per_device_train_batch_size = None
        self.per_device_eval_batch_size = None
        self.num_train_epochs = None
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.save_strategy = 'no'
        self.load_best_model_at_end = False
        self.push_to_hub = False
        self.eval_accumulation_steps = 1
        self.predict_with_generate = True
        self.max_token_length = 128
        self.bos_instruction = None
        self.delim_instruction = None
        self.eos_instruction = None
        self.test_input = None
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args()) 
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', help='train/eval/cli', type=str, required=True)
        parser.add_argument('-model_checkpoint', help='Huggingface Model Path', type=str, required=True)
        parser.add_argument('-inst_type', help='Decides if InstructABSA1 or InstructABSA2', type=int)
        parser.add_argument('-experiment_name', help='Name of experiment', type=str)
        parser.add_argument('-task', help='ate/atsc/joint', type=str)
        parser.add_argument('-output_dir', type=str)
        parser.add_argument('-id_tr_data_path', type=str)
        parser.add_argument('-id_te_data_path', type=str)
        parser.add_argument('-set_instruction_key', type=int, default=1)
        parser.add_argument('-ood_tr_data_path', type=str)
        parser.add_argument('-ood_te_data_path', type=str)
        parser.add_argument('-output_path', type=str)
        parser.add_argument('-sample_size', help='For sampling fraction of train data', default=1.0, type=float)
        parser.add_argument('-evaluation_strategy', help='no/epoch/steps', default='epoch', type=str)
        parser.add_argument('-learning_rate', help='learning rate', default=5e-5, type=float)
        parser.add_argument('-per_device_train_batch_size', help='per_device_train_batch_size', default=16, type=int)
        parser.add_argument('-per_device_eval_batch_size', help='per_device_eval_batch_size', default=16, type=int)
        parser.add_argument('-num_train_epochs', help='num_train_epochs', default=4, type=float)
        parser.add_argument('-weight_decay', help='Weight decay', default=0.01, type=float)
        parser.add_argument('-warmup_ratio', help='Warmup Ratio', default=0.1, type=float)
        parser.add_argument('-save_strategy', help='no/epoch/steps', default='no', type=str)
        parser.add_argument('-eval_accumulation_steps', help='Eval gradient accumulation steps', default=1, type=int)
        parser.add_argument('-predict_with_generate', help='Predict with generate', default=True, type=bool)
        parser.add_argument('-max_token_length', help='Sets maximum token output length', default=128, type=bool)
        parser.add_argument('-test_input', help='The input review to test', type=str)
        return parser