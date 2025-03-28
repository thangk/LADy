import random, logging, pickle, pandas as pd, numpy as np, os, string

import nltk
import torch

from octis.models.model import *
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from .mdl import AbstractAspectModel
from cmn.review import Review

class Nrl(AbstractAspectModel):
    def __init__(self, octis_mdl, naspects, nwords, metrics):
        super().__init__(naspects, nwords)
        self.mdl = octis_mdl
        self.metrics = metrics

    def name(self): return 'octis.' + self.mdl.__class__.__name__.lower()
    def _seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True

    def load(self, path):
        self.mdl_out = load_model_output(f'{path}model.out.npz', vocabulary_path=None, top_words=self.nwords)
        self.mdl = pd.read_pickle(f'{path}model')
        self.dict = pd.read_pickle(f'{path}model.dict')
        self.cas = pd.read_pickle(f'{path}model.perf.cas')
        self.perplexity = pd.read_pickle(f'{path}model.perf.perplexity')

    def _create_ds(self, reviews_train, reviews_valid, output):
        if not os.path.isdir(output): os.makedirs(output)
        df_train = Review.to_df(reviews_train, w_augs=False)
        df_train['part'] = 'train'
        df_valid = Review.to_df(reviews_valid, w_augs=False)
        df_valid['part'] = 'val'
        df = pd.concat([df_train, df_valid])
        df.to_csv(f'{output}/corpus.tsv', sep='\t', encoding='utf-8', index=False, columns=['text', 'part'], header=None)

    def train(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output):

        dataset = Dataset()
        try: dataset.load_custom_dataset_from_folder(f'{output}corpus')
        except:
            self._create_ds(reviews_train, reviews_valid, f'{output}corpus')
            dataset.load_custom_dataset_from_folder(f'{output}corpus')
        self.dict = dataset.get_vocabulary()
        self.mdl.hyperparameters.update(settings)
        self.mdl.hyperparameters.update({'num_topics': self.naspects})
        self.mdl.hyperparameters.update({'save_dir': None})#f'{output}model'})

        if 'bert_path' in self.mdl.hyperparameters.keys(): self.mdl.hyperparameters['bert_path'] = f'{output}corpus/'
        self.mdl.use_partitions = True
        self.mdl.update_with_test = True
        self.mdl_out = self.mdl.train_model(dataset, top_words=self.nwords)

        save_model_output(self.mdl_out, f'{output}model.out')
        # octis uses '20NewsGroup' as default corpus when no text passes! No warning?!
        self.cas = Coherence(texts=dataset.get_corpus(), topk=self.nwords, measure='u_mass', processes=settings['ncore']).score(self.mdl_out)
        pd.to_pickle(self.cas, f'{output}model.perf.cas')
        pd.to_pickle(self.dict, f'{output}model.dict')
        pd.to_pickle(self.mdl, f'{output}model')
        pd.to_pickle(self.perplexity, f'{output}model.perf.perplexity')

    def get_aspect_words(self, aspect_id, nwords):
        word_list = self.mdl_out['topics'][aspect_id]
        probs = []
        for w in word_list: probs.append(self.mdl_out['topic-word-matrix'][aspect_id][self.dict.index(w)])
        return list(zip(word_list, probs))

    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        reviews_test_ = []; reviews_aspects = []
        
        print(f"Processing {len(reviews_test)} reviews in NRL model...")
        
        # Check if this is an implicit dataset by looking at the first review's aos structure
        is_implicit = False
        if reviews_test and hasattr(reviews_test[0], 'implicit'):
            is_implicit = any(reviews_test[0].implicit)
            print(f"NRL: Detected implicit dataset: {is_implicit}")
        
        for r in reviews_test:
            try:
                # Get aspects based on whether this is implicit or explicit
                if is_implicit:
                    r_aspects = []
                    for sent_idx, sent in enumerate(r.aos):
                        sent_aspects = []
                        if r.implicit[sent_idx]:  # For implicit sentences
                            for aos_tuple in sent:
                                # For implicit aspects, the 4th element is the aspect term
                                if len(aos_tuple) >= 4 and aos_tuple[3] != 'NULL':
                                    sent_aspects.append(aos_tuple[3])  # Add the aspect term
                        else:  # For explicit sentences
                            for aos_tuple in sent:
                                # Extract aspect words from indices
                                if aos_tuple[0]:  # If aspect indices exist
                                    aspect_words = [r.sentences[sent_idx][idx] for idx in aos_tuple[0]]
                                    sent_aspects.extend(aspect_words)
                        r_aspects.append(sent_aspects)
                else:
                    # Extract aspects for explicit dataset using get_aos
                    r_aspects = []
                    for sent in r.get_aos():
                        sent_aspects = []
                        for aos_tuple in sent:
                            # Handle both 3-element and 4-element tuples
                            if len(aos_tuple) >= 3:  # Basic check that we have at least (a,o,s)
                                a, o, s = aos_tuple[:3]  # Extract first three elements
                                sent_aspects.extend([w for w in a if w is not None])
                        r_aspects.append(sent_aspects)
                
                # Skip reviews with no aspects
                if not r_aspects or all(len(sent) == 0 for sent in r_aspects):
                    print(f"NRL: Skipping review {r.id} with empty aspects")
                    continue
                
                # Add this review for processing
                if random.random() < h_ratio: r_ = r.hide_aspects()
                else: r_ = r
                reviews_aspects.append(r_aspects)
                reviews_test_.append(r_)
                
            except Exception as e:
                print(f"NRL: Error processing review {r.id}: {str(e)}")
                continue
        
        # Check if we have any reviews to process
        if not reviews_test_:
            print("NRL: Error: No valid reviews to process. All reviews were skipped due to errors or empty aspects.")
            # Return an empty list of pairs
            return []
        
        # Process the valid reviews
        print(f"NRL: Successfully processed {len(reviews_test_)} reviews out of {len(reviews_test)}")
        
        try:
            #like in ctm (isinstance(self, CTM))
            if 'bert_model' in self.mdl.hyperparameters: 
                _, test, input_size = self.mdl.preprocess(self.mdl.vocab, [], test=[r.get_txt() for r in reviews_test_], bert_model=self.mdl.hyperparameters['bert_model'])
            # like in neurallda isinstance(self, NeuralLDA)
            else: 
                _, test, input_size = self.mdl.preprocess(self.mdl.vocab, [], test=[r.get_txt() for r in reviews_test_])
            
            test = self.mdl.inference(test)

            reviews_pred_aspects = [test['test-topic-document-matrix'][:, rdx] for rdx, _ in enumerate(reviews_test_)]
            pairs = []
            for i, r_pred_aspects in enumerate(reviews_pred_aspects):
                r_pred_aspects = [[(j, v) for j, v in enumerate(r_pred_aspects)]]
                pairs.extend(list(zip(reviews_aspects[i], self.merge_aspects_words(r_pred_aspects, self.nwords))))
            
            return pairs
        except Exception as e:
            print(f"NRL: Error in model inference: {str(e)}")
            # For debugging purposes
            print(f"NRL: Number of reviews to process: {len(reviews_test_)}")
            return []
