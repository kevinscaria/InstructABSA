import torch
import numpy as np
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)

# class DatasetLoader:
#     def __init__(self, experiment_name, rest_train_df=None, rest_test_df=None, lap_train_df=None, 
#                  lap_test_df=None, train_df=None, test_df=None, sample_size = 1) -> None:
#         if 'restaurants' in experiment_name:
#             self.train_df_id = rest_train_df.sample(frac = sample_size, random_state = 1999)
#             self.test_df_id = rest_test_df
#             self.train_df_ood = lap_train_df.sample(frac = sample_size, random_state = 1999)
#             self.test_df_ood = lap_test_df
#             self.train_df = None
#             self.test_df = None
#             self.train_df_id_name = 'restaurants_train.csv'
#             self.test_df_id_name = 'restaurants_test.csv'
#             self.train_df_ood_name = 'laptops_train.csv'
#             self.test_df_ood_name = 'laptops_test.csv'
#         elif 'laptops' in experiment_name:
#             self.train_df_id = lap_train_df.sample(frac = sample_size, random_state = 1999)
#             self.test_df_id = lap_test_df
#             self.train_df_ood = rest_train_df.sample(frac = sample_size, random_state = 1999)
#             self.test_df_ood = rest_test_df
#             self.train_df = None
#             self.test_df = None
#             self.train_df_id_name = 'laptops_train.csv'
#             self.test_df_id_name = 'laptops_test.csv'
#             self.train_df_ood_name = 'restaurants_train.csv'
#             self.test_df_ood_name = 'restaurants_test.csv' 
#         elif 'combined' in experiment_name:
#             self.train_df = pd.concat([rest_train_df, lap_train_df]).reset_index(drop=True).sample(frac = sample_size, random_state = 1999)
#             self.test_df = pd.concat([rest_test_df, lap_test_df]).reset_index(drop=True)
#             self.train_df_id = None
#             self.test_df_id = None
#             self.train_df_ood = None
#             self.test_df_ood = None
#             self.train_name = 'combined_train.csv'
#             self.test_name = 'combined_test.csv'
#         else:
#             self.train_df = train_df.sample(frac = sample_size)
#             self.test_df = test_df
#             self.train_df_id = None
#             self.test_df_id = None
#             self.train_df_ood = None
#             self.test_df_ood = None
#             self.train_name = f'{experiment_name}_train.csv'
#             self.test_name = f'{experiment_name}_test.csv'

#     def set_data_for_training_semeval(self, tokenize_function, experiment_name, remove_cols = []):
#         """
#         Create the training and test dataset as huggingface datasets format.
#         """
#         # Define train and test sets
#         if experiment_name.find('restaurants')!= -1 or experiment_name.find('laptops')!=-1:
#             indomain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_id), 'test': Dataset.from_pandas(self.test_df_id)})
#             for rem_col in remove_cols:
#                 indomain_dataset = indomain_dataset.remove_columns(rem_col)

#             other_domain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_ood), 'test': Dataset.from_pandas(self.test_df_ood)})
#             other_domain_dataset = other_domain_dataset.remove_columns(remove_cols)

#             indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
#             other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
#             try:  
#                 indomain_tokenized_datasets = indomain_tokenized_datasets.remove_columns(["text", "__index_level_0__"])
#                 other_domain_tokenized_dataset = other_domain_tokenized_dataset.remove_columns(["text", "__index_level_0__"])
#             except:
#                 indomain_tokenized_datasets = indomain_tokenized_datasets.remove_columns(["text"])
#                 other_domain_tokenized_dataset = other_domain_tokenized_dataset.remove_columns(["text"])
#             return indomain_dataset, indomain_tokenized_datasets, other_domain_dataset, other_domain_tokenized_dataset

#         else:
#             dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df), 'test': Dataset.from_pandas(self.test_df)})
#             for rem_col in remove_cols:
#                 dataset = dataset.remove_columns(rem_col)

#             tokenized_dataset = dataset.map(self.tokenize_function_inputs, batched=True)

#             try:  
#                 tokenized_dataset = tokenized_dataset.remove_columns(["text", "__index_level_0__"])
#             except:
#                 tokenized_dataset = tokenized_dataset.remove_columns(["text"])

#             return dataset, tokenized_dataset, None, None        


class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        #Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer


    def get_labels(self, tokenized_dataset, trained_model_path=None, predictor = None, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        if not predictor:
            ft_model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_path)

            # Prediction args
            pred_args = Seq2SeqTrainingArguments(
                output_dir = './',
                do_train = False,
                do_predict = True,
                per_device_eval_batch_size = batch_size 
            )

            # Initialize prediction trainer
            predictor = Seq2SeqTrainer(
                        model = ft_model, 
                        args = pred_args
                        )

        output_ids = predictor.predict(test_dataset=tokenized_dataset[sample_set]).predictions
        trainer_outputs = [i.replace('<pad>', '').replace('</s>', '').lstrip().rstrip() for i in self.tokenizer.batch_decode(output_ids)]
        return trainer_outputs

    def get_csv_filename(self, experiment_id, ):
        if experiment_id in ['restaurants', 'laptops']:
            return [self.train_df_id_name, self.test_df_id_name, self.train_df_ood_name, self.test_df_ood_name]
        else:
            return [self.train_name, self.test_name]



class T5Classifier:
    def __init__(self, model_checkpoint, experiment_id=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)


    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample
    
    def set_data_for_training_semeval(self, experiment_id):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        if experiment_id in ['restaurants', 'laptops']:
            indomain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_id), 'test': Dataset.from_pandas(self.test_df_id)})
            remove_cols = ["aspectCategories", "sentenceId", "aspectTerms", "raw_text"]
            if 'aspect' in self.train_df_id.columns:
                remove_cols+=['aspect']
            indomain_dataset = indomain_dataset.remove_columns(remove_cols)

            other_domain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_ood), 'test': Dataset.from_pandas(self.test_df_ood)})
            other_domain_dataset = other_domain_dataset.remove_columns(remove_cols)

            indomain_tokenized_datasets = indomain_dataset.map(self.tokenize_function_inputs, batched=True)
            other_domain_tokenized_dataset = other_domain_dataset.map(self.tokenize_function_inputs, batched=True)

            try:  
                indomain_tokenized_datasets = indomain_tokenized_datasets.remove_columns(["text", "__index_level_0__"])
                other_domain_tokenized_dataset = other_domain_tokenized_dataset.remove_columns(["text", "__index_level_0__"])
            except:
                indomain_tokenized_datasets = indomain_tokenized_datasets.remove_columns(["text"])
                other_domain_tokenized_dataset = other_domain_tokenized_dataset.remove_columns(["text"])

            return indomain_dataset, other_domain_dataset, indomain_tokenized_datasets, other_domain_tokenized_dataset

        else:
            indomain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df), 'test': Dataset.from_pandas(self.test_df)})
            remove_cols = ["aspectCategories", "sentenceId", "aspectTerms", "raw_text"]
            if 'aspect' in self.train_df.columns:
                remove_cols+=['aspect']
            indomain_dataset = indomain_dataset.remove_columns(remove_cols)
            indomain_tokenized_datasets = indomain_dataset.map(self.tokenize_function_inputs, batched=True)

            try:  
                indomain_tokenized_datasets = indomain_tokenized_datasets.remove_columns(["text", "__index_level_0__"])
            except:
                indomain_tokenized_datasets = indomain_tokenized_datasets.remove_columns(["text"])

            return indomain_dataset, indomain_tokenized_datasets
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = TrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator 
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, trained_model_path=None, predictor = None, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        if not predictor:
            ft_model = T5ForConditionalGeneration.from_pretrained(trained_model_path)

            # Prediction args
            pred_args = TrainingArguments(
                output_dir = './',
                do_train = False,
                do_predict = True,
                per_device_eval_batch_size = batch_size,   
            )

            # Initialize prediction trainer
            predictor = Trainer(
                        model = ft_model, 
                        args = pred_args, 
                        data_collator = self.data_collator 
                        )
        pred_proba = predictor.predict(test_dataset=tokenized_dataset[sample_set]).predictions[0]
        output_ids = np.argmax(pred_proba, axis=2)
        trainer_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return trainer_outputs, pred_proba


    def get_csv_filename(self, experiment_id, ):
        if experiment_id in ['restaurants', 'laptops']:
            return [self.train_df_id_name, self.test_df_id_name, self.train_df_ood_name, self.test_df_ood_name]
        else:
            return [self.train_name, self.test_name]



# class T5JointTask:
#     def __init__(self, model_checkpoint, experiment_id=None, rest_train_df=None, rest_test_df=None, lap_train_df=None, 
#                  lap_test_df=None, train_df=None, test_df=None, valid_df=None, sample_size = 1):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
#         self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

#     def tokenize_function_inputs(self, sample):
#         """
#         Udf to tokenize the input dataset.
#         """
#         model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)

#         # Setup the tokenizer for targets
#         labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
        
#     def train(self, tokenized_datasets, **kwargs):
#         """
#         Train the generative model.
#         """

#         #Set training arguments
#         args = Seq2SeqTrainingArguments(
#             **kwargs
#         )

#         # Define trainer object
#         trainer = Seq2SeqTrainer(
#             self.model,
#             args,
#             train_dataset=tokenized_datasets["train"],
#             eval_dataset=tokenized_datasets["test"],
#             tokenizer=self.tokenizer,
#             data_collator=self.data_collator,
#         )
#         print("Trainer device:", trainer.args.device)

#         # Finetune the model
#         torch.cuda.empty_cache()
#         print('\nModel training started ....')
#         trainer.train()

#         # Save best model
#         trainer.save_model()
#         return trainer

#     def get_labels(self, tokenized_dataset, trained_model_path=None, predictor = None, batch_size =4, sample_set = 'train'):
#         """
#         Get the predictions from the trained model.
#         """
#         if not predictor:
#             ft_model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_path)

#             # Prediction args
#             pred_args = Seq2SeqTrainingArguments(
#                 output_dir = './',
#                 do_train = False,
#                 do_predict = True,
#                 per_device_eval_batch_size = batch_size,
#             )

#             # Initialize prediction trainer
#             predictor = Seq2SeqTrainer(
#                         model = ft_model, 
#                         args = pred_args, 
#                         data_collator = self.data_collator 
#                         )

#         output_ids = predictor.predict(test_dataset=tokenized_dataset[sample_set]).predictions
#         trainer_outputs = [i.replace('<pad>', '').replace('</s>', '').lstrip().rstrip() for i in self.tokenizer.batch_decode(output_ids)]
#         return trainer_outputs

#     def get_csv_filename(self, experiment_id, ):
#         if experiment_id in ['restaurants', 'laptops']:
#             return [self.train_df_id_name, self.test_df_id_name, self.train_df_ood_name, self.test_df_ood_name]
#         else:
#             return [self.train_name, self.test_name]