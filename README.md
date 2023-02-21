# InstructABSA: Instruction learning for Aspect Based Sentiment Classification

This repository is the official implementation of the [paper](https://arxiv.org/abs/2302.08624). As part of our approach we show the efficacy of instruction tuning language modelling on downstream ABSA subtasks and surpasses previous SOTA by significant margin.

This Tk-Instruct model is further insturction-tuned on downstream tasks of ABSA as follows:

For Aspect Term Extraction (ATE) subtask: $$A_i = LM_{Inst}(Inst,S_i)$$

For Aspect Term Sentiment Classification (ATSC) subtask: $$sp_i^k = LM_{Inst}(Inst,S_i, a_i^k)$$

And to model Joint Task: $$[A_i, SP_i] = LM_{Inst}(Inst,S_i)$$


## Aspect Term Extraction

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. There are two ways this can be done. The first one is through the CLI commands which is shown below:

```shell
python run_model.py
```

To run the same using the InstructABSA module, the steps are described below:
```python
from InstructABSA.data_prep import DataLoader
```

## Aspect Term Sentiment Classification

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. There are two ways this can be done. The first one is through the CLI commands which is shown below:

```shell
python run_model.py
```

To run the same using the InstructABSA module, the steps are described below:
```python
from InstructABSA.data_prep import DataLoader
```

## Joint Tasks

The ATE models can be trained from scratch or alternatively can be used to run inference on your datasets directly. There are two ways this can be done. The first one is through the CLI commands which is shown below:

```shell
python run_model.py
```

To run the same using the InstructABSA module, the steps are described below:
```python
from InstructABSA.data_prep import DataLoader
```

