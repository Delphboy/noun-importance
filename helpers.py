import json

import numpy as np


def load_weights(path: str) -> dict:
    with open(path) as f:
        weights = json.load(f)
    return weights


def save_weights(weights: dict, name: str):
    with open(name, "w") as f:
        json.dump(weights, f)


def convert_key_to_int(weights: dict) -> dict:
    w2i = {v: k for k, v in load_sg_data().items()}
    converted_weights = {}
    for key in weights:
        converted_weights[w2i[key]] = weights[key]

    return converted_weights


def remove_leading_space(sentence: str) -> str:
    return sentence[1 : len(sentence)]


def remove_trailing_space(sentence: str) -> str:
    return sentence[: len(sentence) - 1]


def preprocess_sentence(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = sentence.replace(",", "")
    sentence = sentence.replace(".", "")
    sentence = sentence.replace("'", "")
    sentence = sentence.replace('"', "")
    sentence = sentence.replace("\n", "")

    while "  " in sentence:
        sentence = sentence.replace("  ", " ")

    if sentence[0] == " ":
        sentence = remove_leading_space(sentence)

    if sentence[len(sentence) - 1] == " ":
        sentence = remove_trailing_space(sentence)

    return sentence


def load_sg_data() -> dict:
    return np.load("coco_pred_sg_rela.npy", allow_pickle=True).item()["i2w"]


def load_sg_data_words() -> list:
    return list(load_sg_data().values())
