import argparse
import random

import spacy

from corpus import generate_test_sentences, get_sentences
from helpers import (
    convert_key_to_int,
    load_sg_data_words,
    preprocess_sentence,
    save_weights,
)
from stats import (
    get_average,
    get_maximum,
    get_minimum,
    normalise_weights,
    plot_histogram,
    print_top_k,
)

random.seed(0)
nlp = spacy.load("en_core_web_sm")

sg_data_words = load_sg_data_words()


DETECTABLE_OBJECTS = []
for word in sg_data_words:
    doc = nlp(word)
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            DETECTABLE_OBJECTS.append(word)


def calculate_local_noun_importance(num_nouns: int, position: int) -> float:
    val = num_nouns - (position / num_nouns)
    return val if val > 0 else 1


def calculate_global_noun_importance(corpus) -> dict:
    weights = {}
    occurrences = {}

    for sentence in corpus:
        sentence = preprocess_sentence(sentence)
        doc = nlp(sentence)

        # count the number of nouns in the sentence
        nouns = []
        for token in doc:
            if token.text in DETECTABLE_OBJECTS:
                nouns.append(token.text)

        for i in range(len(nouns)):
            token = nouns[i]
            if token in DETECTABLE_OBJECTS:
                if token not in weights:
                    weights[token] = calculate_local_noun_importance(len(nouns), i + 1)
                    occurrences[token] = 1
                else:
                    weights[token] += calculate_local_noun_importance(len(nouns), i + 1)
                    occurrences[token] += 1

    avg_weights = {}
    for noun in weights:
        avg_weights[noun] = weights[noun] / occurrences[noun]

    sorted_keys = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)
    return {key[0]: key[1] for key in sorted_keys}


def calcualte_noun_pairings(noun_weights):
    noun_pairings = {}
    for noun in noun_weights:
        for noun2 in noun_weights:
            noun_pairings[f"({noun}, {noun2})"] = (
                noun_weights[noun] + noun_weights[noun2]
            )
    return noun_pairings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate",
        help="generate new test captions from a 10 percent coco split",
        action="store",
    )
    parser.add_argument(
        "--corpus",
        help="path to corpus file",
        default="corpus/test-captions.txt",
    )
    parser.add_argument(
        "--statistics",
        help="calculate statistics for the corpus",
        action="store_true",
    )
    parser.add_argument(
        "--convert_key_to_int",
        help="convert keys to integers (converts noun keys to Visual Genome IDs)",
        action="store_true",
    )
    parser.add_argument(
        "--pair_nouns",
        help="calculate noun pairings",
        action="store_true",
    )
    parser.add_argument(
        "--normalise",
        help="normalise the weights",
        action="store_true",
    )
    parser.add_argument(
        "--save",
        help="save the weights to a file",
        default="noun_weights.txt",
        action="store",
    )

    args = parser.parse_args()
    if args.generate:
        generate_test_sentences(int(args.generate))
        exit(0)
    if args.corpus:
        corpus = get_sentences(args.corpus)

    weights = calculate_global_noun_importance(corpus)

    if args.convert_key_to_int:
        weights = convert_key_to_int(weights)

    if args.pair_nouns:
        weights = calcualte_noun_pairings(weights)

    if args.normalise:
        weights = normalise_weights(weights)

    if args.save:
        save_weights(weights, args.save)

    if args.statistics:
        print_top_k(weights, 10)
        plot_histogram(weights)
        print(f"Average: {get_average(weights)}")
        print(f"Maximum: {get_maximum(weights)}")
        print(f"Minimum: {get_minimum(weights)}")
