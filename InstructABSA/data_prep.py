from datasets import Dataset
from datasets.dataset_dict import DatasetDict


class DatasetLoader:
    def __init__(self, train_df_id=None, test_df_id=None, val_df_id=None, 
                 train_df_ood=None, test_df_ood=None, val_df_ood=None, sample_size = 1):
        
        self.train_df_id = train_df_id.sample(frac = sample_size, random_state = 1999) if train_df_id is not None else train_df_id
        self.test_df_id = test_df_id
        self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood
        self.val_df_id = val_df_id
        self.val_df_ood = val_df_ood

    def reconstruct_strings(self, df, col):
        """
        Reconstruct strings to dictionaries when loading csv/xlsx files.
        """
        reconstructed_col = []
        for text in df[col]:
            if text != '[]' and isinstance(text, str):
                text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '').split(", '")
                req_list = []
                for idx, pair in enumerate(text):
                    splitter = ': ' if ': ' in pair else ':'
                    if idx%2==0:
                        reconstructed_dict = {} 
                        reconstructed_dict[pair.split(splitter)[0].replace("'", '')] = pair.split(splitter)[1].replace("'", '')
                    else:
                        reconstructed_dict[pair.split(splitter)[0].replace("'", '')] = pair.split(splitter)[1].replace("'", '')
                        req_list.append(reconstructed_dict)
            else:
                req_list = text
            reconstructed_col.append(req_list)
        df[col] = reconstructed_col
        return df

    def extract_rowwise_aspect_polarity(self, df, on, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        try:
            df.iloc[0][on][0][key]
        except:
            df = self.reconstruct_strings(df, on)

        df['len'] = df[on].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[on, 'record_idx']].apply(lambda x : (x[0][x[1]][key], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('',''), axis=1)
        df['polarity'] = df['aspect'].apply(lambda x: x[-1])
        df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df
    
    def extract_rowwise_aspect_opinions(self, df, aspect_col, opinion_col, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        df['len'] = df[aspect_col].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[aspect_col, 'record_idx']].apply(lambda x : x[0][x[1]][key] if len(x[0]) != 0 else '', axis=1)
        df['opinion_term'] = df[[opinion_col, 'record_idx']].apply(lambda x : x[0][x[1]][key] if len(x[0]) != 0 else '', axis=1)
        df['aspect'] = df['aspect'].apply(lambda x: ' '.join(x))
        df['opinion_term'] = df['opinion_term'].apply(lambda x: ' '.join(x))
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df

    def create_data_in_ate_format(self, df, key, text_col, aspect_col, bos_instruction = '', 
                    eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col)
        df['labels'] = df[aspect_col].apply(lambda x: ', '.join([i[key] for i in x]))
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df

    def create_data_in_atsc_format(self, df, on, key, text_col, aspect_col, bos_instruction = '', 
                    delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        df = self.extract_rowwise_aspect_polarity(df, on=on, key=key, min_val=1)
        df['text'] = df[[text_col, aspect_col]].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        df = df.rename(columns = {'polarity': 'labels'})
        return df

    def create_data_in_aspe_format(self, df, key, label_key, text_col, aspect_col, bos_instruction = '', 
                                         eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col)
        df['labels'] = df[aspect_col].apply(lambda x: ', '.join([f"{i[key]}:{i[label_key]}" for i in x]))
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df
    
    def create_data_in_aooe_format(self, df, aspect_col, opinion_col, key, text_col, 
                               bos_instruction = '', delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        df = self.extract_rowwise_aspect_opinions(df, aspect_col=aspect_col, opinion_col=opinion_col, key=key, min_val=1)
        df['text'] = df[[text_col, 'aspect']].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        df = df.rename(columns = {'opinion_term': 'labels'})
        return df
    
    def create_data_in_aope_format(self, df, key, text_col, aspect_col, opinion_col,
                                         bos_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        df['labels'] = df[[aspect_col, opinion_col]].apply(lambda x: ', '.join([f"{' '.join(i[key])}:{' '.join(j[key])}" for i, j in zip(x[0], x[1])]), axis=1)
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df
    
    def create_data_in_aoste_format(self, df, key, label_key, text_col, aspect_col, opinion_col,
                                         bos_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        label_map = {'POS':'positive', 'NEG':'negative', 'NEU':'neutral'}
        df['labels'] = df[[aspect_col, opinion_col]].apply(lambda x: ', '.join([f"{' '.join(i[key])}:{' '.join(j[key])}:{label_map[i[label_key]]}" for i, j in zip(x[0], x[1])]), axis=1)
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df
    
    def set_data_for_training_semeval(self, tokenize_function):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        dataset_dict_id, dataset_dict_ood = {}, {}

        if self.train_df_id is not None:
            dataset_dict_id['train'] = Dataset.from_pandas(self.train_df_id)
        if self.test_df_id is not None:
            dataset_dict_id['test'] = Dataset.from_pandas(self.test_df_id)
        if self.val_df_id is not None:
            dataset_dict_id['validation'] = Dataset.from_pandas(self.val_df_id)
        if len(dataset_dict_id) > 1:
            indomain_dataset = DatasetDict(dataset_dict_id)
            indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
        else:
            indomain_dataset = {}
            indomain_tokenized_datasets = {}

        if self.train_df_ood is not None:
            dataset_dict_ood['train'] = Dataset.from_pandas(self.train_df_ood)
        if self.test_df_ood is not None:
            dataset_dict_ood['test'] = Dataset.from_pandas(self.test_df_ood)
        if self.val_df_ood is not None:
            dataset_dict_ood['validation'] = Dataset.from_pandas(self.val_df_ood)
        if len(dataset_dict_id) > 1:
            other_domain_dataset = DatasetDict(dataset_dict_ood)
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
        else:
            other_domain_dataset = {}
            other_domain_tokenized_dataset = {}

        return indomain_dataset, indomain_tokenized_datasets, other_domain_dataset, other_domain_tokenized_dataset
        