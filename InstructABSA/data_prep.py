import os
import pandas as pd
import xml.etree.ElementTree as ET


class ModelReadyData:
    def __init__(self, ):
        pass


    def parse_xml(self, path, save_csv = False, output_path = None, overwrite = False):
        """
        Extract the information requried from the XML based data source.
        """
        file_name = path.split('/')[-1]
        file_name = file_name.replace('xml', 'csv')

        if not os.path.isfile(os.path.join(output_path, file_name)) or\
            (os.path.isfile(os.path.join(output_path, file_name)) and overwrite):
            if save_csv and not output_path:
                raise Exception("You have provided the option to save csv, but no output path. Enter the output_path to dump the csv.")

            else:    
                reviews, idx_list = [], []
                tree = ET.parse(path)
                root = tree.getroot()
                for child in root:
                    idx_list.append(child.attrib['id'])

                for sentence in root.findall("sentence"):
                    entry = {}
                    aspect_terms = []
                    aspect_categories = []
                    
                    if sentence.find("aspectTerms"):
                        for aspect_term in sentence.find("aspectTerms").findall("aspectTerm"):
                            aspect_terms.append((aspect_term.get("term"), aspect_term.get("polarity")))
                            
                    if sentence.find("aspectCategories"):
                        for aspect_category in sentence.find("aspectCategories").findall("aspectCategory"):
                            aspect_categories.append((aspect_category.get("category"), aspect_category.get("polarity")))
                            
                    entry["raw_text"] = sentence[0].text
                    entry["aspectTerms"] = aspect_terms
                    entry["aspectCategories"] = aspect_categories
                    reviews.append(entry)

                df = pd.DataFrame(reviews)
                df['sentenceId'] = idx_list
                df['aspectTerms'] = df['aspectTerms'].apply(lambda x:  [{'term':i[0], 'polarity':i[1]} for i in x] if len(x)>0 else [{'term':'noaspectterm', 'polarity':'none'}])
                df['aspectCategories'] = df['aspectCategories'].apply(lambda x:  [{'category':i[0], 'polarity':i[1]} for i in x] if len(x)>0 else [{'category':'noaspectcategory', 'polarity':'none'}])
                df = df[['sentenceId', 'raw_text', 'aspectTerms', 'aspectCategories']]

                if save_csv:
                    df.to_csv(os.path.join(output_path, file_name), index=False)
                    return df
        else:
            print(f'File {os.path.join(output_path, file_name)} has already been extracted!!')
            return os.path.join(output_path, file_name)


    def reconstruct_strings(self, df, col):
        """
        Reconstruct strings to dictionaries when loading csv files.
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


    def extract_rowwise_aspect_polarity(self, df, on, by, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        try:
            df.iloc[0][on][0][by]
        except:
            self.reconstruct_strings(df, on)

        df['len'] = df[on].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[on, 'record_idx']].apply(lambda x : (x[0][x[1]][by], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('',''), axis=1)
        df['polarity'] = df['aspect'].apply(lambda x: x[-1])
        df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df


    def create_data_in_ate_format(self, df, on, text_col, aspect_col, rename_target_col, bos_instruction = '', 
                    eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        try:
            df.iloc[0][aspect_col][0][on]
        except:
            self.reconstruct_strings(df, aspect_col)
        df[rename_target_col] = df[aspect_col].apply(lambda x: ', '.join([i[on] for i in x]))
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df


    def create_data_in_atsc_format(self, df, text_col, aspect_col, rename_target_col, bos_instruction = '', 
                    delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        df['text'] = df[[text_col, aspect_col]].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        df = df.rename(columns = {'polarity': rename_target_col})
        return df


    def create_data_in_ate_atsc_format(self, df, on, text_col, aspect_col, rename_target_col, bos_instruction = '', 
                                        eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        try:
            df.iloc[0][aspect_col][0][on]
        except:
            self.reconstruct_strings(df, aspect_col)
        df[rename_target_col] = df[aspect_col].apply(lambda x: ', '.join([f"{i['term']}:{i['polarity']}" for i in x]))
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df