# InstructABSA: Instruction learning for Aspect Based Sentiment Classification

This repository is the official implementation of the [paper](https://arxiv.org/abs/2302.08624). As part of our approach we show the efficacy of instruction tuned language models. This approach surpasses previous SOTA on downstream ABSA subtasks by significant margin.

## Dataset Requirements
This section describes the format of the data required for the training and evaluation of the datasets using our approach. For all subtasks, the field names should match exactly as shown and have the same datatypes. The fields to be present in the raw dataset are as follows:

-- ```raw_text```: This is the reviews section (str)

-- ```aspectTerms```: This is the set of aspect terms and their polarities to be present as a list of dictionaries. Each dictionary will have atleast two keys with the one of the key ```term``` and the value which is an aspect in the corresponding sentence. The second key will be ```polarity``` and its value is the polarity for corresponding aspect. (```[{'term':'aspect1', 'polarity':'polarity1'}, ...]```)

>**Note**
>When creating the dataset in this fashion and saving it, .xlsx/.csv format will convert the aspectTerms column into ```string/text``` format. But the package will handle that when loading the dataset file. 

An example dataset is shown below:
| raw_text  | aspectTerms |
| ------------- | ------------- |
| The cab ride was amazing but the service was pricey  | [{'term':'cab ride', 'polarity':'positive'}, {'term':'service', 'polarity':'negative'}]  |
| I ordered the Barbeque Pizza | []  |

## Aspect Term Extraction

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. There are two ways this can be done. The first one is through the CLI commands which is shown below:

```shell
python run_model.py
```

To run the same using the InstructABSA module, the steps are described below:
```python
from InstructABSA.utils import T5Generator
```

## Aspect Term Sentiment Classification

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. There are two ways this can be done. The first one is through the CLI commands which is shown below:

```shell
python run_model.py
```

To run the same using the InstructABSA module, the steps are described below:
```python
from InstructABSA.utils import T5Classifier
```

## Joint Tasks

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. There are two ways this can be done. The first one is through the CLI commands which is shown below:

```shell
python run_model.py
```

To run the same using the InstructABSA module, the steps are described below:
```python
from InstructABSA.utils import T5Generator
```

