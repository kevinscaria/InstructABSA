# InstructABSA: Instruction learning for Aspect Based Sentiment Classification

This repository is the official implementation of the [paper](https://arxiv.org/abs/2302.08624). As part of our approach we show the efficacy of instruction tuned language models. This approach surpasses previous SOTA on downstream ABSA subtasks by significant margin.

## Dataset Requirements
This section describes the format of the data required for the training and evaluation of the datasets using our approach. For all subtasks, the field name should match exactly as shown and have the same datatypes. The fields to be present in the raw dataset are as follows:
```python
-- raw_text: This is the reviews section (str)
```

```python
-- aspectTerms: This is the set of aspect terms and their polarities ([{'term':'asoect1', 'polarity':'polarity'}...]
```

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

