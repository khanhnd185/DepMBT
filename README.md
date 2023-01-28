# Multimodal Bottleneck Transformer for Depression Recognition

## Introduction

Google Researchers introduces a transformer-based fusion model named [Multimodal Bottleneck Transformer](https://ai.googleblog.com/2022/03/multimodal-bottleneck-transformer-mbt.html) (MBT) for fusing visual and audio features. Traditional methods try to concatenate the sequences of two or more embeddings from different modalities in temporal dimension. It consumes a lot of resources (time, memory, and computation) due to the quaradtic complexity of attention mechanism. MBT devises a new special token named bottleneck tokens. It is intermediary token to transfer information between two modalities, instead of paying attention to the whole concatenated sequence.

![MBT explaination from Google Blog](https://raw.githubusercontent.com/khanhnd185/my-blog/my-pages/_posts/images/mbt/mbt.png)

## Model

My implementation uses Pytorch framework. Some code from [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) is borrowed. Some functions are imported from [timm](https://github.com/rwightman/pytorch-image-models)

The model is evaluated in 2 benchmarks: [D-Vlog Dataset](https://ojs.aaai.org/index.php/AAAI/article/view/21483) and [EATD-Corpus](https://arxiv.org/abs/2202.08210). The objective is depression recognition.

## Result

Evaluation metrics on EATD-Corpus Test set
| Model | F1 | Precision | Recall |
| --- | --- | ---| ---|
| Multimodal LSTM | 0.57 | 0.49 | 0.67 |
| Baseline | 0.71 | 0.62 | 0.84 |
| TAMFN | 0.75 | 0.69 | 0.85 |
| CE-DepMBT | 0.77 | 0.86 | 0.83 |


Evaluation metrics on D-VLog Test set
| Model | F1 | Precision | Recall |
| --- | --- | ---| ---|
| Baseline | 0.635 | 0.654 | 0.656 |
| TAMFN | 0.658 | 0.660 | 0.665 |
| CE-DepMBT | 0.646 | 0.657 | 0.660 |
| SupCon-DepMBT | 0.663 | 0.675 | 0.671 |

