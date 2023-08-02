# Noun Importance
Given a set of sentences, how important are the nouns? How important are the noun to noun pairings?

This repository provides a naïve implementation of applying weights to a set of nouns. It was developed for exploritory work with the [COCO Dataset](https://cocodataset.org/). However, it can be run on any text file that contains sentences seperated by lines.

## Dependencies

- Python 3.9.16
- See `requirements.txt`

```bash
git clone git@github.com:Delphboy/noun-importance.git
cd noun-importance

# Create a python environment using preferred method

python3 -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

- `--corpus`: Path the the corpus file
- `--generate <int>`: Generate <int> sized corpus from captions within the 10% COCO split
- `--statistics`: Whether or not to write out some basic statistics to the console
- `--convert_key_to_int`: Whether or not to convert the noun to the Visual Genome object id given in [coco_pred_sg_rela.npy](coco_pred_sg_rela.npy)
- `--pair_nouns`: Whether or not to produce weights of all possible noun pairs
- `--normalise`: Whether or not to normalise the weights
- `--save <loc>`: Saves the final weight file to <loc> *Needs to be a JSON file*

### Examples

Generate a corpus of 100 captions
```bash
python3 main.py --generate 100
```

Generate weights for test captions and print statistics
```bash
python3 main.py --corpus corpus/test-captions.txt --statistics
```

Typical run
```bash
python3 main.py --corpus corpus/test-captions.txt --convert_key_to_int --pair_nouns --statistics --save weights/example.json
```
