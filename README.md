# MLGCN
Jiang Baoxing, Xu Guangtao & Liu Peiyu Aspect-level sentiment classification via location enhanced aspect-merged graph convolutional networks. J Supercomput (2023). https://doi.org/10.1007/s11227-022-05002-4

## Requirements
* Python 3.9
* PyTorch 1.12
* SpaCy 3.3.1
* numpy 1.21.5
* entmax 1.0
* argparse 1.4.0
* scikit-learn 1.0.2

## Usage

* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and unzip `glove.840B.300d.txt` into `./glove/`.
* Prepare data for models, run the code [prepare_data.py](./prepare_data.py)
```bash
pthon ./prepare_data.py
```
* If you want to train with Dependency based Location-aware transformation in command, optional arguments could be found in [train_dep.py](/.train_dep.py)
```bash
python ./train_dep.py --dataset lap14 --num_epoch 100 --learning_rate 0.001 --repeats 5
```
* If you want to train with SE-attention based Location-aware transformation in command, optional arguments could be found in [train_se.py](/.train_se.py)
```bash
python ./train_se.py --dataset lap14 --num_epoch 100 --learning_rate 0.001 --repeats 5
```
* Training with the script file [run.sh](./run.sh)
```bash
bash run.sh
```

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper.

## Note
* Code of this repo heavily relies on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and [ASGCN](https://github.com/GeneZC/ASGCN)
