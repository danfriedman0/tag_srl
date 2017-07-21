# tag_srl: Semantic Role Labeling with Tree-adjoining Grammars

This is an SRL model based on the syntax-agnostic model of Marcheggiani et al (2017). See https://arxiv.org/abs/1701.02593 and https://github.com/diegma/neural-dep-srl. The model also supports using supertags from a tree-adjoining grammar in addition to part of speech tags.

## Dependencies

* Python 2.7
* TensorFlow 1.0

## Required data

You need your own copy of the CoNLL-2009 dataset to train the model. To use all the training options, you need the to add the following files:

```
data/
  conll09/
	  gold/
		  dev.tag
			dev.txt
			ood.tag
			ood.txt
			test.tag
			test.txt
			train.tag
			train.txt
		pred/
		  dev.tag
			dev.txt
			ood.tag
			ood.txt
			test.tag
			test.txt
			train.tag
			train.txt			
```

`data/conll09/gold/*.txt` is the file from the CoNLL-2009 release, so for example `data/conll09/gold/dev.txt` is the same as `CoNLL2009-ST-English-development.txt` in the CoNLL release.

`data/conll09/pred/*.txt` is the same as `data/conll09/gold/*.txt`, but with predicted predicate senses in the 14th column instead of gold standard predicates. We use [mate-tools](https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki) to do predicate identification. You need to download the SRL pipeline (`srl-4.31.tgz`) and the English model file (linked from [here](https://code.google.com/archive/p/mate-tools/wikis/Models.wiki)). Then you should be able to run mate-tools on each text file data/conll09/gold/.

mate-tools also does semantic role labeling, but the files in data/conll09/pred/ should have gold argument labels. So once you've run mate-tools, you can generate the appropriate files by running, for example,
```
paste <(cut -f -13 data/conll09/gold/dev.txt) <(cut -f 14 mate-tools_dev.txt) <(cut -f 15- data/conll09/gold/dev.txt) > data/conll09/pred/dev.txt
```
and similarly for the rest of hte files.

`data/conll09/*/*.tag` is a file in CoNLL format with an additional column for supertags. You can do this in a couple of ways:

* Download the UD_STAG extraction code from https://github.com/forrestdavis/UD_STAG. Run the code by moving to the SRL directory and running `srl_trees.py <file.txt>` for each file, and moving the output file (`file.tag`) to the appropriate directory.
* If you don't want to run the supertag experiments, you should be able to just copy each .txt file to a .tag file, and I think the code will still run. Just make sure not to use the `--use_supertags` flag.

You also need a copy of the pre-trained word embeddings used in the paper. Download the `sskip.100.vectors` file from https://github.com/diegma/neural-dep-srl and put it in `data/embeddings/sskip.100.vectors`.


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
