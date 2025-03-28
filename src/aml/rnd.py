import gensim, pandas as pd, random
from typing import List, Tuple

import numpy as np

from .mdl import AbstractAspectModel, ModelCapabilities, ModelCapability
from cmn.review import Review

PairType = Tuple[List[str], List[Tuple[str, float]]]

class Rnd(AbstractAspectModel):
    name = 'rnd'
    capabilities: ModelCapabilities = {'aspect_detection'}

    def __init__(self, n_aspects: int, n_words: int):
        self.n_aspects = n_aspects
        self.nw = n_words
        # Create dummy aspect words for predictions, consistent across runs
        self.dummy_pred_words = {i: f"rnd_aspect_{i}" for i in range(self.n_aspects)}
        print(f"Initialized Random model with n_aspects={n_aspects}")

    def load(self, path):
        self.dict = gensim.corpora.Dictionary.load(f'{path}model.dict')
        pd.to_pickle(self.cas, f'{path}model.perf.cas')
        pd.to_pickle(self.perplexity, f'{path}model.perf.perplexity')

    def infer(self, review, doctype):
        review_ = super(Rnd, self).preprocess(doctype, [review])
        return [[(0, 1)] for r in review_]

    def get_aspect_words(self, aspect_id, nwords): return [(i, 1) for i in random.sample(self.dict.token2id.keys(), min(nwords, len(self.dict)))]

    def train(self, reviews_train: List[Review], reviews_valid: List[Review], settings: dict, doctype: str, no_below_above: Tuple[int, float], output: str=None) -> None:
        """Random model requires no training."""
        print("Random model: No training required.")
        pass

    def infer_batch(self, reviews_test: List[Review], h_ratio: float, doctype: str, output:str=None) -> List[PairType]:
        """
        Perform inference for a batch of reviews.
        For each review, extracts the ground truth aspect words and generates random predictions.
        """
        pairs: List[PairType] = []
        print(f"Random model: Inferring aspects for {len(reviews_test)} reviews...")

        for r in reviews_test:
            # 1. Extract Ground Truth Aspect Words
            true_aspect_words = set()
            try:
                # Check if review has sentences and aspects-over-sentence data
                if r.sentences and r.aos:
                    # Assuming the first sentence contains the relevant tokens and aspects
                    if len(r.sentences) > 0:
                        tokens = r.sentences[0]
                        # Check if there's aspect data for the first sentence
                        if len(r.aos) > 0 and r.aos[0]:
                            for aos_instance in r.aos[0]: # aos_instance is like ([idx], [], sentiment, 'NULL' or aspect_term)
                                # Check if it's an explicit aspect term (not 'NULL') and has indices
                                if len(aos_instance) >= 4 and aos_instance[3] != 'NULL':
                                    token_indices = aos_instance[0]
                                    # Validate indices before accessing tokens
                                    if all(0 <= idx < len(tokens) for idx in token_indices):
                                        aspect_phrase = " ".join(tokens[idx] for idx in token_indices)
                                        true_aspect_words.add(aspect_phrase)
                                    else:
                                        print(f"Warning (Review {r.id}): Invalid token indices {token_indices} for {len(tokens)} tokens.")
                                # For implicit aspects ('NULL'), we don't add them here as ground truth words
            except Exception as e:
                print(f"Error extracting true aspects for review {r.id}: {e}")
                # Decide how to handle errors: skip review, add empty list, etc.

            # 2. Generate Random Aspect Predictions
            pred_aspects: List[Tuple[str, float]] = []
            try:
                # Sample n_aspects unique random indices from the available dummy words
                num_to_sample = min(self.n_aspects, len(self.dummy_pred_words))
                if num_to_sample > 0:
                    pred_indices = random.sample(list(self.dummy_pred_words.keys()), num_to_sample)
                    # Assign equal probability/weight
                    weight = 1.0 / num_to_sample
                    pred_aspects = [(self.dummy_pred_words[idx], weight) for idx in pred_indices]
                    # Sort predictions alphabetically by dummy aspect name for consistency
                    pred_aspects.sort()
            except Exception as e:
                 print(f"Error generating random predictions for review {r.id}: {e}")
                 # Handle error, maybe return empty predictions

            # Append the pair: (list of ground truth words, list of predicted dummy words with weights)
            pairs.append((list(true_aspect_words), pred_aspects))

        print(f"Random model: Finished inference for {len(pairs)} reviews.")
        return pairs

    def quality(self, qtype: str) -> float:
        """Random model has no meaningful quality measure during training."""
        return 0.0

    def save(self, output: str = None):
        """Random model has no state to save."""
        print("Random model: No model state to save.")
        pass

    def load(self, input: str = None):
        """Random model has no state to load."""
        print("Random model: No model state to load.")
        pass

