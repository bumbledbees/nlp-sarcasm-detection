from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain, filterfalse, pairwise
import json
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

from nltk import FreqDist
import nltk.corpus as corpus
from nltk.tokenize import word_tokenize

DATASET_URL = 'https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/refs/heads/master/Sarcasm_Headlines_Dataset.json'

@dataclass(frozen=True)
class Headline:
    text: str
    tokens: tuple[str]
    is_sarcastic: bool
    url: str

# this function was taken from AST1 lol
def deapostraphize(tokens: Iterable[str]) -> str:
    it_tokens = enumerate(tokens)
    n_tokens = len(tokens)

    for idx, token in it_tokens:
        if idx < (n_tokens - 1) and "'" in tokens[idx + 1]:
            _, token_next = next(it_tokens)
            token += token_next
        yield token

def apostrophe_tokenize(text: str) -> tuple[str]:
    return tuple(deapostraphize(word_tokenize(text)))

def parse_dataset(raw_json: str) -> list[Headline]:
    headlines: list[Headline] = []

    for line in raw_json.splitlines():
        entry = json.loads(line)
        headline = Headline(text=entry['headline'],
                            #tokens=tuple(word_tokenize(entry['headline'])),
                            tokens=apostrophe_tokenize(entry['headline']),
                            is_sarcastic=bool(entry['is_sarcastic']),
                            url=entry['article_link'])
        headlines.append(headline)

    return headlines

def main() -> int:
    # has the dataset been downloaded?
    data_path = Path(Path(urlparse(DATASET_URL).path).name)
    if data_path.exists():
        print(f'Loading dataset from {data_path}...')
        with open(data_path, 'r') as file:
            raw_data = file.read()
    else:
        print(f"Downloading dataset from '{DATASET_URL}'...")
        with urlopen(DATASET_URL) as resp:
            raw_data = resp.read().decode('utf-8')

        with open(data_path, 'w') as file:
            file.write(raw_data)

    dataset = parse_dataset(raw_data)
    is_sarcastic = lambda d: d.is_sarcastic
    onion_data = list(filter(is_sarcastic, dataset))
    hpost_data = list(filterfalse(is_sarcastic, dataset))

    stopwords = set(corpus.stopwords.words('english'))
    stopwords.update({"'s", "n't", "'re"})
    punct = {',', "'", ':', '.', '?', '!', '(', ')', '-', '--', '$', '%', '@',
             '#', '^', '*', '/', '\\'}
    stopwords.update(punct)

    onion_vocab = FreqDist(chain.from_iterable(d.tokens for d in onion_data))
    hpost_vocab = FreqDist(chain.from_iterable(d.tokens for d in hpost_data))
    # filter stopwords / punctuation
    for vocab in onion_vocab, hpost_vocab:
        for word in stopwords:
            vocab.pop(word, None)

    iter_bigrams = lambda x: pairwise(chain.from_iterable(d.tokens for d in x))
    onion_bigrams = FreqDist(iter_bigrams(onion_data))
    hpost_bigrams = FreqDist(iter_bigrams(hpost_data))
    # filter bigrams where both words are stopwords / punctuation
    for vocab in onion_bigrams, hpos_bigrams:
        for bigram in vocab:
            if all(w in stopwords for w in bigram):
                vocab.pop(bigram)

    def report_topn(n: int, fdist: FreqDist):
        for idx, (word, count) in enumerate(fdist.most_common(n)):
            print(f'{idx + 1}) "{word}": {count}')
        print()

    n = 10
    print(f'Top {n} most common words (The Onion):')
    report_topn(n, onion_vocab)
    print(f'Top {n} most common words (The Huffington Post):')
    report_topn(n, hpost_vocab)
    print(f'Top {n} most common bigrams (The Onion):')
    report_topn(n, onion_bigrams)
    print(f'Top {n} most common bigrams (The Huffington Post):')
    report_topn(n, hpost_bigrams)

if __name__ == main():
    sys.exit(main())
