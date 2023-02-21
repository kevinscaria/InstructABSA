import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
from .data_prep import DatasetLoader
from .utils import T5Generator, T5Classifier

from .config import Config

try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

# Set Global Values
config = Config()
experiment_name = config.experiment_name
model_checkpoint = config.model_checkpoint
model_out_path = config.output_dir
model_out_path = os.path.join(model_out_path, f"{model_checkpoint}-{experiment_name}", "checkpoints")

id_tr_data_path = config.id_tr_data_path
ood_tr_data_path = config.ood_tr_data_path
id_te_data_path = config.id_te_data_path
ood_te_data_path = config.ood_te_data_path
tr_data_path = config.tr_data_path
te_data_path = config.te_data_path
print('Loaded data...')
print('Model dump path: ', model_out_path)

# Load the data
id_tr_df, id_te_df = None, None
ood_tr_df, ood_te_df = None, None
tr_df, te_df = None, None

if id_tr_data_path is not None:
    id_tr_df = pd.read_csv(id_tr_data_path)
if id_te_data_path is not None:
    id_te_df = pd.read_csv(id_te_data_path)
if ood_tr_data_path is not None:
    ood_tr_df = pd.read_csv(ood_tr_data_path)
if ood_te_data_path is not None:
    ood_te_df = pd.read_csv(ood_te_data_path)
if tr_data_path is not None:
    tr_df = pd.read_csv(tr_data_path)
if id_tr_data_path is not None:
    te_df = pd.read_csv(id_tr_data_path)

# Training arguments
training_args = {
                'output_dir': config.output_dir,
                'evaluation_strategy': config.evaluation_strategy,
                'learning_rate': config.learning_rate,
                'per_device_train_batch_size': config.per_device_train_batch_size,
                'per_device_eval_batch_size': config.per_device_eval_batch_size,
                'num_train_epochs': config.num_train_epochs,
                'weight_decay': config.weight_decay,
                'warmup_ratio': config.warmup_ratio,
                'save_strategy': config.save_strategy,
                'load_best_model_at_end': config.load_best_model_at_end,
                'push_to_hub': config.push_to_hub,
                'eval_accumulation_steps': config.eval_accumulation_steps,
                'predict_with_generate': config.predict_with_generate, 
                'use_mps_device': use_mps
            }

print(training_args)

# Create T5 utils object
if config.task not in 'atsc':
    t5_exp = T5Generator(model_checkpoint)
else:
    t5_exp = T5Classifier(model_checkpoint)


# Tokenize Datasets
bos_instruction_id = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
Positive example 1-
input: I charge it at night and skip taking the cord with me because of the good battery life.
output: battery life
Positive example 2-
input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
output: features, iChat, Photobooth, garage band
Negative example 1-
input: Speaking of the browser, it too has problems.
output: browser
Negative example 2-
input: The keyboard is too slick.
output: keyboard
Neutral example 1-
input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
output: battery
Neutral example 2-
input: Nightly my computer defrags itself and runs a virus scan.
output: virus scan
Now complete the following example-
input: """

bos_instruction_ood = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
Positive example 1-
input: With the great variety on the menu , I eat here often and never get bored.
output: menu
Positive example 2- 
input: Great food, good size menu, great service and an unpretensious setting.
output: food, menu, service, setting
Negative example 1-
input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
output: toast, mayonnaise, bacon, ingredients, plate
Negative example 2-
input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
output: seats
Neutral example 1-
input: I asked for seltzer with lime, no ice.
output: seltzer with lime
Neutral example 2-
input: They wouldnt even let me finish my glass of wine before offering another.
output: glass of wine
Now complete the following example-
input: """

delim_instruction = ''

eos_instruction = ' \noutput:'

# Define function to load datasets and tokenize datasets
loader = DatasetLoader(experiment_name, id_tr_df, id_te_df, ood_tr_df, ood_te_df, tr_df, te_df, config.sample_size)

if config.task == 'ate':
    if loader.train_df_id is not None:
        train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
    if loader.test_df_id is not None:
        test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    if loader.train_df_ood is not None:
        train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
    if loader.test_df_ood is not None:
        test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    if loader.train_df is not None:
        train_df = loader.create_data_in_ate_format(loader.train_df, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
    if loader.test_df is not None:
        test_df = loader.create_data_in_ate_format(loader.test_df, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

elif config.task == 'atsc':
    if loader.train_df_id is not None:
        train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
    if loader.test_df_id is not None:
        test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
    if loader.train_df_ood is not None:
        train_df_ood = loader.create_data_in_atsc_format(loader.train_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
    if loader.test_df_ood is not None:
        test_df_ood = loader.create_data_in_atsc_format(loader.test_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
    if loader.train_df is not None:
        train_df = loader.create_data_in_atsc_format(loader.train_df, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
    if loader.test_df is not None:
        test_df = loader.create_data_in_atsc_format(loader.test_df, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)

elif config.task == 'joint':
    if loader.train_df_id is not None:
        train_df_id = loader.create_data_in_joint_task_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
    if loader.test_df_id is not None:
        test_df_id = loader.create_data_in_joint_task_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    if loader.train_df_ood is not None:
        train_df_ood = loader.create_data_in_joint_task_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
    if loader.test_df_ood is not None:
        test_df_ood = loader.create_data_in_joint_task_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    if loader.train_df is not None:
        train_df = loader.create_data_in_joint_task_format(loader.train_df, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
    if loader.test_df is not None:
        test_df = loader.create_data_in_joint_task_format(loader.test_df, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

tokenize_function = t5_exp.tokenize_function_inputs
id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(tokenize_function, experiment_name)
print(id_tokenized_ds)
print(ood_tokenzed_ds)

# Train Model
model_trainer = t5_exp.train(id_tokenized_ds, **training_args)

# Model inference
print('Getting model from path: ', model_out_path)

id_tr_pred_labels = t5_exp.get_labels(predictor = model_trainer, tokenized_dataset = id_tokenized_ds, sample_set = 'train')
id_te_pred_labels = t5_exp.get_labels(predictor = model_trainer, tokenized_dataset = id_tokenized_ds, sample_set = 'test')

if ood_tokenzed_ds is not None:
    ood_tr_pred_labels = t5_exp.get_labels(predictor = model_trainer, tokenized_dataset = ood_tokenzed_ds, sample_set = 'train')
    ood_te_pred_labels = t5_exp.get_labels(predictor = model_trainer, tokenized_dataset = ood_tokenzed_ds, sample_set = 'test')