import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier, Evaluator
from InstructABSA.config import Config
from instructions import InstructionsHandler

try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

# Set Global Values
config = Config()
instruct_handler = InstructionsHandler()
instruct_handler.load_instruction_set2()
print('Task: ', config.task)

if config.experiment_name is not None:
    print('Experiment Name: ', config.experiment_name)
    model_checkpoint = config.model_checkpoint
    model_out_path = config.output_dir
    model_out_path = os.path.join(model_out_path, config.task, f"{model_checkpoint}-{config.experiment_name}")
else:
    model_checkpoint = config.model_checkpoint

print('Mode set to: ', 'training' if config.mode == 'train' else ('inference' if config.mode == 'inference' \
                                                                  else 'Individual sample inference'))

# Load the data
id_tr_data_path = config.id_tr_data_path
ood_tr_data_path = config.ood_tr_data_path
id_te_data_path = config.id_te_data_path
ood_te_data_path = config.ood_te_data_path

if config.mode != 'cli':
    id_tr_df = pd.read_csv(id_tr_data_path)
    id_te_df = pd.read_csv(id_te_data_path)
    ood_tr_df,  ood_te_df = None, None

    if ood_tr_data_path is not None:
        ood_tr_df = pd.read_csv(ood_tr_data_path)
    if ood_te_data_path is not None:
        ood_te_df = pd.read_csv(ood_te_data_path)
    print('Loaded data...')
else:
    print('Running inference on input: ', config.test_input)

# Training arguments
training_args = {
                'output_dir': model_out_path,
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

# Create T5 model object
if config.set_instruction_key:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'

if config.task == 'ate':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ate[indomain]
    if ood_tr_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
    eos_instruction = instruct_handler.ate['eos_instruct']
if config.task == 'atsc':
    t5_exp = T5Classifier(model_checkpoint)
    bos_instruction_id = instruct_handler.atsc[indomain]
    if ood_tr_data_path is not None:
        bos_instruction_ood = instruct_handler.atsc[outdomain]
    delim_instruction = instruct_handler.atsc['delim_instruct']
    eos_instruction = instruct_handler.atsc['eos_instruct']
if config.task == 'joint':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.joint[indomain]
    if ood_tr_data_path is not None:
        bos_instruction_ood = instruct_handler.joint[outdomain]
    eos_instruction = instruct_handler.joint['eos_instruct']

if config.mode != 'cli':
    # Define function to load datasets and tokenize datasets
    loader = DatasetLoader(id_tr_df, id_te_df, ood_tr_df, ood_te_df, config.sample_size)
    if config.task == 'ate':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

    elif config.task == 'atsc':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_atsc_format(loader.train_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_atsc_format(loader.test_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)

    elif config.task == 'joint':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_joint_task_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_joint_task_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_joint_task_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_joint_task_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

    # Tokenize dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

    if config.mode == 'train':
        # Train model
        model_trainer = t5_exp.train(id_tokenized_ds, **training_args)
        print('Model saved at: ', model_out_path)
    elif config.mode == 'eval':
        # Get prediction labels
        print('Model loaded from: ', model_out_path)
        evals = Evaluator()
        id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'train', trained_model_path = model_out_path)
        id_te_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'test', trained_model_path = model_out_path)
        id_tr_df = pd.DataFrame(id_ds['train'])[['text', 'labels']]
        id_te_df = pd.DataFrame(id_ds['test'])[['text', 'labels']]
        id_tr_df['pred_labels'], id_te_df['pred_labels'] = id_tr_pred_labels, id_te_pred_labels
        # Dump output
        id_tr_df.to_csv(config.output_path, index=False)
        id_te_df.to_csv(config.output_path, index=False)

        print('*****Train Metrics*****')
        precision, recall, f1 = evals.get_metrics(id_tr_df['labels'], id_tr_pred_labels)
        print('Precision: ', precision)
        print('Recall: ', precision)
        print('F1-Score: ', precision)

        print('*****Test Metrics*****')
        precision, recall, f1 = evals.get_metrics(id_te_df['labels'], id_te_pred_labels)
        print('Precision: ', precision)
        print('Recall: ', precision)
        print('F1-Score: ', precision)

        if ood_tokenzed_ds:
            ood_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenzed_ds, sample_set = 'train', trained_model_path = model_out_path)
            ood_te_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenzed_ds, sample_set = 'test', trained_model_path = model_out_path)
            ood_tr_df = pd.DataFrame(ood_ds['train'])[['text', 'labels']]
            ood_te_df = pd.DataFrame(ood_ds['test'])[['text', 'labels']]
            ood_tr_df['pred_labels'], ood_te_df['pred_labels'] = ood_tr_pred_labels, ood_te_pred_labels
            # Dump output
            ood_tr_df.to_csv(config.output_path, index=False)
            ood_te_df.to_csv(config.output_path, index=False)

            print('*****Train Metrics - OOD*****')
            precision, recall, f1 = evals.get_metrics(ood_tr_df['labels'], ood_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)

            print('*****Test Metrics - OOD*****')
            precision, recall, f1 = evals.get_metrics(ood_te_df['labels'], ood_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
else:
    print('Model loaded from: ', model_checkpoint)
    print('HERE')
    if config.task == 'atsc':
        config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
        model_input = bos_instruction_id + config.test_input + f'. The aspect term is: {aspect_term}' + eos_instruction
    else:
        model_input = bos_instruction_id + config.test_input + eos_instruction
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids)
    print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))