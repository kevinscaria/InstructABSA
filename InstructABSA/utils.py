import torch
import numpy as np
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)


class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
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


class T5Classifier:
    def __init__(self, model_checkpoint):
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
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
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
            pred_args = Seq2SeqTrainingArguments(
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
        return trainer_outputs
    

class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics(self, ):
        total_pred = 0
        total_gt = 0
        tp = 0
        for gt, pred in zip(self.y_true, self.y_pred):
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred+=len(pred_list)
            total_gt+=len(gt_list)
            for gt_val in gt_list:
                for pred_val in pred_list:
                    if pred_val in gt_val:
                        tp+=1
        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r) 