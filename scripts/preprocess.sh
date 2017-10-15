#!/bin/bash
# Preprocesses a CoNLL09 dataset for use in SRL model.
# The argument is a language, e.g.
#   ./scripts/preprocess.sh eng

LANG=$1

echo "Extracting supertags..."
python scripts/extract.py 1 data/$LANG/conll09/
mkdir -p data/$LANG/stags/
mv data/$LANG/conll09/stags.* data/$LANG/stags/
mkdir -p data/$LANG/conll09/gold/
mv data/$LANG/conll09/*_* data/$LANG/conll09/gold


echo "Extracting predicates..."
cut -f 14 data/$LANG/conll09/dev.txt > data/$LANG/conll09/gold/dev_predicates.txt
cut -f 14 data/$LANG/conll09/test.txt > data/$LANG/conll09/gold/test_predicates.txt
cut -f 14 data/$LANG/conll09/train.txt > data/$LANG/conll09/gold/train_predicates.txt
if [ -e data/$LANG/conll09/ood.txt ]
then
    cut -f 14 data/$LANG/conll09/ood.txt > data/$LANG/conll09/gold/ood_predicates.txt
fi


echo "Creating vocabs..."
mkdir -p data/$LANG/vocab/

# Takes a column of words and outputs a list of of sorted 'word count' pairs
to_vocab () {
    cat $1 |
      grep -v "^\s*$" |
      sort |
      uniq -c |
      sed 's/^ *//g' |
      sort -bnr |
      awk ' { t = $1; $1 = $2; $2 = t; print } '
}

# Words
cut -f 2 data/$LANG/conll09/train.txt |  # Get words
  tr '[:upper:]' '[:lower:]' |         # Lowercase
  sed -E 's/[0-9]+\n/<NUM>/' |         # Replace integers with NUM token
  sed -E 's/[0-9]*\.[0-9]+/<FLOAT>/' | # Replace floats with FLOAT token
  to_vocab > data/$LANG/vocab/words.txt

# Parts of speech
cut -f 6 data/$LANG/conll09/train.txt |
  to_vocab > data/$LANG/vocab/pos.txt

# Model 1 supertags
cat data/$LANG/conll09/gold/train_stags_model1.txt |
  to_vocab > data/$LANG/vocab/stags.model1.txt

# Predicates (full)
cat data/$LANG/conll09/gold/train_predicates.txt |
  to_vocab > data/$LANG/vocab/predicates.txt

# Predicate lemmas
cat data/$LANG/conll09/gold/train_predicates.txt |
  sed 's/\([a-z]*\)\..*/\1/' | # Ignore the number part of the predicate
  to_vocab > data/$LANG/vocab/lemmas.txt

# Regular predicted lemmas
cut -f 4 data/$LANG/conll09/train.txt |
  to_vocab > data/$LANG/vocab/plemmas.txt

# Semantic role labels
cut -f 15- data/eng/conll09/train.txt |  # Get the label columns
  awk '{gsub(/\t/, "\n"); print }' | # Flatten to one column
  to_vocab > data/$LANG/vocab/labels.txt
