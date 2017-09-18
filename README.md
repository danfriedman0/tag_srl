# tag_srl: Semantic Role Labeling with Supertags

This is a model for semantic dependency parsing based on the syntax-agnostic model of Marcheggiani et al (2017). See https://arxiv.org/abs/1701.02593 and https://github.com/diegma/neural-dep-srl. This model achieves state-of-the-art results by a) using a better regularized LSTM with highway layers, as in He et al (20017), and b) incorporating supertags extracted from dependency trees.

## Dependencies

* Python 2.7
* TensorFlow 1.0

## Required data

You need your own copy of the CoNLL-2009 dataset to train the model. Here are the steps to prepare the data for a given language (here, English):
1. From the `tag_srl` directory, create the directories `data/eng/conll09/` ("eng" for English).
2. Add the CoNLL text files to the directory and name them `train.txt`, `dev.txt`, `test.txt`, and `ood.txt`, if there is an out-of-domain dataset.
3. Run `./scripts/preprocess.sh eng` to extract supertags from the data and generate vocab files.
4. You also need a copy of the pre-trained word embeddings used in the paper. Download the `sskip.100.vectors` file from https://github.com/diegma/neural-dep-srl and put it in `data/eng/embeddings/sskip.100.vectors`. (Instructions for other languages forthcoming.)
5. To replicate our results, you really need to use predicted predicates and supertags and put them in `data/eng/conll09/pred/` with names	`dev_predicates.txt`, `dev_stags_model1.txt`, etc.
  a. We used [mate-tools](https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki) to get predicted predicates. You need to download the SRL pipeline (`srl-4.31.tgz`) and the English model file (linked from [here](https://code.google.com/archive/p/mate-tools/wikis/Models.wiki)).
	b. Contact me to get predicted supertags.


## Training

Once you have all of the code, you can train a model by running (from the root directory)
```
python model/train.py
```
See `python model/train.py --help` for all of the possible options. The default arguments are all the same as the best hyperparemeters from Marcheggiani et al (2017), but I found that the model performs even better if you add dropout between layers of the bidirectional LSTM (`python model/train.py --dropout 0.5`).

Training on one Nvidia Tesla K80 GPU, with a batch size of 100, the model took around 13 minutes per epoch, and our best models converged after 3-6 hours of training.


## Testing

Once you've trained a model, the trained model will be saved in a directory in `output/models/`. To test the model, run
```
python model/test.py output/models/model_name test_split
```
`test_split` should be dev, test, ood (out-of-domain), or train.

## Organization

The code for this project is organized as follows:

1. `model/train.py` parses command line arguments, resolves filepaths, and then calls `utils/vocab.py` to get vocabulary objects for words, parts of speech, etc. A `Vocab` object has methods for mapping words to integer ids and back.
2. Then `model/train.py` calls `model/srl.py` to initialize the `SRL_Model` object. `SRL_Model` builds a neural network using generic neural network functions from `model/layers.py` and `model/lstm.py`.
3. `model/srl.py` uses functions from `util/conll_io.py` and `util/data_loader.py` to load CoNLL sentences into the model.
	 a. `util/conll_io.py` defines methods for reading and writing CoNLL-09 format sentences and storing them in a convenient format, along with predicted predicates and supertags.
	 b. `util/data_loader.py` converts lists of CoNLL sentences to batches of integer ids that can be fed to the neural network.
4. The `SRL_Model` also has methods for running a training or testing epoch. In a testing epoch, the model writes its predictions to a file (usually `output/predictions/{model_name}.txt`), in CoNLL format, so it can be evaluated with the CoNLL evaluation script.
5. After each epoch, `model/train.py` calls the CoNLL-provided perl evaluation script (from a python wrapper in `eval/eval.py`) and decides whether or not to stop early.
6. `model/test.py` just rebuilds a trained model and runs a testing epoch with the specified data.
7. The predicate disambiguation directory, `model/disamb/`, is organized in the same way, with one file, `model/disamb/disamb.py`, to build the network and run training and testing epochs, and one file each for training and testing the model.