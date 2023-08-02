import json
import random

from helpers import preprocess_sentence

random.seed(0)


def get_sentences(file: str = "corpus/test-captions.txt") -> list:
    with open(file) as f:
        captions = f.readlines()
    return captions


def generate_test_sentences(num_captions: int = 1000) -> None:
    with open("corpus/dataset_coco_10.json") as json_file:
        data = json.load(json_file)

    corpus = []
    for _ in range(num_captions):
        image_id = random.randint(0, len(data["images"]) - 1)
        caption_id = random.randint(0, len(data["images"][0]["sentences"]) - 1)
        caption = data["images"][image_id]["sentences"][caption_id]["raw"]
        caption = preprocess_sentence(caption)
        corpus.append(caption)
    corpus = list(set(corpus))

    with open("corpus/test-captions.txt", "w") as f:
        for caption_id in corpus:
            f.write(caption_id + "\n")
    print(f'Wrote {num_captions} sentences to "captions/test-captions.txt"')
