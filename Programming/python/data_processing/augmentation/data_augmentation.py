from typing import Optional

import nltk
from nltk.corpus import wordnet as wn
from pycorenlp import StanfordCoreNLP

from importer.database.data_types import Post
from importer.database.mongodb import MongodbStorage

nltk.download('wordnet')

nlp = StanfordCoreNLP('http://localhost:9000')
properties_pos = {'annotators': 'pos', 'outputFormat': 'json'}


def _replace_pos(pos_tag: str) -> Optional[str]:
    if pos_tag.startswith("NN"):
        return "n"
    elif pos_tag.startswith("JJ"):
        return "a"
    elif pos_tag.startswith("V"):
        return "v"
    elif pos_tag.startswith("RB"):
        return "r"
    else:
        return None


def get_synonym(post_content: str) -> list:
    output = nlp.annotate(post_content, properties_pos)
    if not isinstance(output, dict):
        return []

    tokens = []
    for sentence in output.get("sentences", []):
        for token in sentence["tokens"]:
            tokens.append((token["originalText"], _replace_pos(token["pos"])))

    augmentations = []
    for index in range(len(tokens)):
        helper(tokens=tokens, index=index, augmentations=augmentations)

    return [" ".join([word[0] for word in sentence]).strip() for sentence in augmentations]


def helper(tokens: list, index: int, augmentations: list):
    if index > len(tokens) - 1:
        return

    token = tokens[index]
    synonyms = []
    if token[1] is not None:
        synsets = wn.synsets(token[0], pos=token[1])
        if len(synsets) > 0:
            synonyms = synsets[0].lemma_names()

    synonyms = list(set([synonym.lower() for synonym in synonyms]))

    synonyms_cleaned = []
    lower = token[0].lower()
    for synonym in synonyms:
        if synonym.lower() != lower:
            synonyms_cleaned.append(synonym)

    del synonyms

    if len(synonyms_cleaned) > 0:
        # Use only the first synonym due to the high number of new posts
        synonym = synonyms_cleaned[0]

        del synonyms_cleaned

        new_tokens = list(tokens)
        new_tokens[index] = (synonym, token[1])
        augmentations.append(new_tokens)
        helper(new_tokens, index=index + 1, augmentations=augmentations)


if __name__ == '__main__':
    db = MongodbStorage()
    reactions_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    count = db.count_posts(filter={})
    for index, post in enumerate(db.iterate_single_post(filter={})):
        assert isinstance(post, Post)

        print("Completed %.2f%%" % ((index + 1) / count * 100))

        sum = 0
        for key in reactions_list:
            sum += post.reactions[key]
        if sum == 0:
            continue

        for index, sentence in enumerate(get_synonym(post.message)):
            data = dict(post.data)
            data[Post.COLL_MESSAGE] = sentence
            data[Post.COLL_POST_ID] += "_{}".format(index)
            new_post = Post(data)
            db.insert_post(new_post)
