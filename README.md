# InstructABSA: Instruction learning for Aspect Based Sentiment Classification

This repository is the official implementation of the [paper](https://arxiv.org/abs/2302.08624). As part of our approach we show the efficacy of instruction tuned language models. This approach surpasses previous SOTA on downstream ABSA subtasks by significant margin.

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

