import os
import pandas as pd
import xml.etree.ElementTree as ET
from datasets import Dataset
from datasets.dataset_dict import DatasetDict


class DatasetLoader:
    def __init__(self, train_df_id, test_df_id, train_df_ood=None, test_df_ood=None, 
                #  train_df=None, test_df=None, 
                 sample_size = 1):
        # if 'restaurants' in experiment_name:
        self.train_df_id = train_df_id.sample(frac = sample_size, random_state = 1999)
        self.test_df_id = test_df_id
        self.train_df_ood = train_df_ood.sample(frac = sample_size, random_state = 1999)
        self.test_df_ood = test_df_ood
            # self.train_df = None
            # self.test_df = None
        #     self.train_df_id_name = 'restaurants_train.csv'
        #     self.test_df_id_name = 'restaurants_test.csv'
        #     self.train_df_ood_name = 'laptops_train.csv'
        #     self.test_df_ood_name = 'laptops_test.csv'
        # elif 'laptops' in experiment_name:
        #     self.train_df_id = lap_train_df.sample(frac = sample_size, random_state = 1999)
        #     self.test_df_id = lap_test_df
        #     self.train_df_ood = rest_train_df.sample(frac = sample_size, random_state = 1999)
        #     self.test_df_ood = rest_test_df
        #     self.train_df = None
        #     self.test_df = None
        #     self.train_df_id_name = 'laptops_train.csv'
        #     self.test_df_id_name = 'laptops_test.csv'
        #     self.train_df_ood_name = 'restaurants_train.csv'
        #     self.test_df_ood_name = 'restaurants_test.csv' 
        # elif 'combined' in experiment_name:
        #     self.train_df = pd.concat([rest_train_df, lap_train_df]).reset_index(drop=True).sample(frac = sample_size, random_state = 1999)
        #     self.test_df = pd.concat([rest_test_df, lap_test_df]).reset_index(drop=True)
        #     self.train_df_id = None
        #     self.test_df_id = None
        #     self.train_df_ood = None
        #     self.test_df_ood = None
        #     self.train_name = 'combined_train.csv'
        #     self.test_name = 'combined_test.csv'
        # else:
        #     self.train_df = train_df.sample(frac = sample_size)
        #     self.test_df = test_df
        #     self.train_df_id = None
        #     self.test_df_id = None
        #     self.train_df_ood = None
        #     self.test_df_ood = None
        #     self.train_name = f'{experiment_name}_train.csv'
        #     self.test_name = f'{experiment_name}_test.csv'

    # def parse_xml(self, path, save_csv = False, output_path = None, overwrite = False):
    #     """
    #     Extract the information requried from the XML based data source.
    #     """
    #     file_name = path.split('/')[-1]
    #     file_name = file_name.replace('xml', 'csv')

    #     if not os.path.isfile(os.path.join(output_path, file_name)) or\
    #         (os.path.isfile(os.path.join(output_path, file_name)) and overwrite):
    #         if save_csv and not output_path:
    #             raise Exception("You have provided the option to save csv, but no output path. Enter the output_path to dump the csv.")

    #         else:    
    #             reviews, idx_list = [], []
    #             tree = ET.parse(path)
    #             root = tree.getroot()
    #             for child in root:
    #                 idx_list.append(child.attrib['id'])

    #             for sentence in root.findall("sentence"):
    #                 entry = {}
    #                 aspect_terms = []
    #                 aspect_categories = []
                    
    #                 if sentence.find("aspectTerms"):
    #                     for aspect_term in sentence.find("aspectTerms").findall("aspectTerm"):
    #                         aspect_terms.append((aspect_term.get("term"), aspect_term.get("polarity")))
                            
    #                 if sentence.find("aspectCategories"):
    #                     for aspect_category in sentence.find("aspectCategories").findall("aspectCategory"):
    #                         aspect_categories.append((aspect_category.get("category"), aspect_category.get("polarity")))
                            
    #                 entry["raw_text"] = sentence[0].text
    #                 entry["aspectTerms"] = aspect_terms
    #                 entry["aspectCategories"] = aspect_categories
    #                 reviews.append(entry)

    #             df = pd.DataFrame(reviews)
    #             df['sentenceId'] = idx_list
    #             df['aspectTerms'] = df['aspectTerms'].apply(lambda x:  [{'term':i[0], 'polarity':i[1]} for i in x] if len(x)>0 else [{'term':'noaspectterm', 'polarity':'none'}])
    #             df['aspectCategories'] = df['aspectCategories'].apply(lambda x:  [{'category':i[0], 'polarity':i[1]} for i in x] if len(x)>0 else [{'category':'noaspectcategory', 'polarity':'none'}])
    #             df = df[['sentenceId', 'raw_text', 'aspectTerms', 'aspectCategories']]

    #             if save_csv:
    #                 df.to_csv(os.path.join(output_path, file_name), index=False)
    #                 return df
    #     else:
    #         print(f'File {os.path.join(output_path, file_name)} has already been extracted!!')
    #         return os.path.join(output_path, file_name)


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
                    if idx%2==0:
                        reconstructed_dict = {}
                        reconstructed_dict[pair.split(': ')[0].replace("'", '')] = pair.split(': ')[1].replace("'", '')
                    else:
                        reconstructed_dict[pair.split(': ')[0].replace("'", '')] = pair.split(': ')[1].replace("'", '')
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
            df[on] = self.reconstruct_strings(df, on)

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

    def create_data_in_joint_task_format(self, df, key, label_key, text_col, aspect_col, bos_instruction = '', 
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
    
    def set_data_for_training_semeval(self, tokenize_function, 
                                    #   experiment_name, remove_cols = []
                                    ):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        # if experiment_name.find('restaurants')!= -1 or experiment_name.find('laptops')!=-1:
        indomain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_id), 'test': Dataset.from_pandas(self.test_df_id)})
        indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)

        if (self.train_df_ood is not None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df_ood), 'test': Dataset.from_pandas(self.test_df_ood)})
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
            return indomain_dataset, indomain_tokenized_datasets, other_domain_dataset, other_domain_tokenized_dataset
        
        return indomain_dataset, indomain_tokenized_datasets, None, None

        # else:
        #     dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df), 'test': Dataset.from_pandas(self.test_df)})
        #     for rem_col in remove_cols:
        #         dataset = dataset.remove_columns(rem_col)

        #     tokenized_dataset = dataset.map(self.tokenize_function_inputs, batched=True)

        #     try:  
        #         tokenized_dataset = tokenized_dataset.remove_columns(["text", "__index_level_0__"])
        #     except:
        #         tokenized_dataset = tokenized_dataset.remove_columns(["text"])

        #     return dataset, tokenized_dataset, None, None     