# InstructABSA: Instruction learning for Aspect Based Sentiment Classification [NAACL - 2024]

## 💥 New Features Coming
- We will be adding LoRA and PTuning based finetuning methods for parameter efficient finetuning.
- We will be updating the task list to add 3 new tasks viz. aspect based opinion extraction (ABPE), aspect opinion pair extraction (AOPE) and aspect opinion sentiment triplet extraction (AOSTE).
- Revisions to paper to include more systematic ablation studies.

## Introduction

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructabsa-instruction-learning-for-aspect/aspect-extraction-on-semeval-2014-task-4-sub-2)](https://paperswithcode.com/sota/aspect-extraction-on-semeval-2014-task-4-sub-2?p=instructabsa-instruction-learning-for-aspect)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructabsa-instruction-learning-for-aspect/aspect-extraction-on-semeval-2014-task-4-sub-1)](https://paperswithcode.com/sota/aspect-extraction-on-semeval-2014-task-4-sub-1?p=instructabsa-instruction-learning-for-aspect)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructabsa-instruction-learning-for-aspect/sentiment-analysis-on-semeval-2014-task-4)](https://paperswithcode.com/sota/sentiment-analysis-on-semeval-2014-task-4?p=instructabsa-instruction-learning-for-aspect)

This repository is the official implementation of the [paper](https://arxiv.org/abs/2302.08624). As part of our approach we show the efficacy of instruction tuned language models. This approach surpasses previous SOTA on downstream ABSA subtasks by significant margin.

## Dataset Requirements
This section describes the format of the data required for the training and evaluation of the datasets using our approach. For all subtasks, the field names should match exactly as shown and have the same datatypes. The fields to be present in the raw dataset are as follows:

-- ```raw_text```: This is the reviews section (str)

-- ```aspectTerms```: This is the set of aspect terms and their polarities to be present as a list of dictionaries. Each dictionary will have atleast two keys with the one of the key ```term``` and the value which is an aspect in the corresponding sentence. The second key will be ```polarity``` and its value is the polarity for corresponding aspect. (```[{'term':'aspect1', 'polarity':'polarity1'}, ...]```)

>**Warning**
>When creating the dataset in this fashion and saving it, ```.xlsx/.csv``` format will convert the aspectTerms column into ```string/text``` format. But the package will handle that when loading the dataset file. 

An example dataset is shown below and also in the [Datasets](https://github.com/kevinscaria/InstructABSA/tree/main/Dataset) folder.
| raw_text  | aspectTerms |
| ------------- | ------------- |
| The cab ride was amazing but the service was pricey  | [{'term':'cab ride', 'polarity':'positive'}, {'term':'service', 'polarity':'negative'}]  |
| I ordered the Barbeque Pizza | [{'term':'noaspectterm', 'polarity':'none'}] |

## Model Checkpoints

All the model weights can be found [here](https://huggingface.co/kevinscaria). The best performing models for each ABSA subtask based on our experiments are presented in the table below:
| Task  | Model Name | Dataset Trained | Model Type | Instruction Configuration |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ATE| kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined | SemEval 2014 Laptops + Restaurants | InstructABSA-2 | Definition + 2 pos + 2 neg + 2 neut examples |
| ATSC| kevinscaria/atsc_tk-instruct-base-def-pos-combined | SemEval 2014 Laptops + Restaurants | InstructABSA-1 | Definition + 2 pos examples |
| Joint Task| kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined | SemEval 2014 Laptops + Restaurants | InstructABSA-2 | Definition + 2 pos + 2 neg + 2 neut examples |

### A sample inference notebook is found [here](https://github.com/kevinscaria/InstructABSA/blob/main/inference_notebook.ipynb).

## Aspect Term Extraction

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. This can be done through CLI (check the [Scripts](https://github.com/kevinscaria/InstructABSA/tree/main/Scripts) folder) or by adapting your code similar to run_model.py. An example shell command to run inference on individual samples is shown below.

A sample notebook for training and evluating ATE can be found [here](https://github.com/kevinscaria/InstructABSA/blob/main/ATE_Training_&_Inference.ipynb).

To evaluate the ATE subtask on a single input using CLI run the following:
```shell
python run_model.py -mode cli -task ate \
-model_checkpoint kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined \
-test_input 'The cab ride was amazing but the service was pricey'
```

## Aspect Term Sentiment Classification

The ATSC models can be trained from scratch or alternatively can be used to run inference on your datasets directly. This can be done through CLI (check the [Scripts](https://github.com/kevinscaria/InstructABSA/tree/main/Scripts) folder) or by adapting your code similar to run_model.py. An example shell command to run inference on individual samples is shown below.

To evaluate the ATSC subtask on a single input using CLI run the following:
```shell
python run_model.py -mode cli -task atsc \
-model_checkpoint kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined \
-test_input 'The ambience was amazing but the waiter was rude|ambience'
```
Note the ```|``` delimiter that is used to pass the aspect term for which the polarity is to be extracted.


## Joint Tasks

The Joint task models can be trained from scratch or alternatively can be used to run inference on your datasets directly. This can be done through CLI (check the [Scripts](https://github.com/kevinscaria/InstructABSA/tree/main/Scripts) folder) or by adapting your code similar to run_model.py. An example shell command to run inference on individual samples is shown below.

A sample notebook for training and evluating Joint Task can be found [here](https://github.com/kevinscaria/InstructABSA/blob/main/JointTask_Training_&_Inference.ipynb).

To evaluate the Joint Task on a single input using CLI run the following:
```shell
python run_model.py -mode cli -task joint \
-model_checkpoint kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined \
-test_input 'The cab ride was amazing but the service was pricey'
```
## Aspect Based Opinion Extraction ⬆️

## Aspect Opinion Pair Extraction ⬆️

## Aspect Opinion Sentiment Triplet Extraction ⬆️

## BibTeX Entry and Citation Info

If you find our work useful, please cite our work: 

```bibtex
article{scaria2023instructabsa,
  title={InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis},
  author={Scaria, Kevin and Gupta, Himanshu and Sawant, Saurabh Arjun and Mishra, Swaroop and Baral, Chitta},
  journal={arXiv preprint arXiv:2302.08624},
  year={2023}
}
```
