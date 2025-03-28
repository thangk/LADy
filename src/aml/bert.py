from typing import Optional, Tuple, List, Dict
import os, re, random
from argparse import Namespace
import pandas as pd

# Define the Aspect_With_Sentiment class needed for type annotations
class Aspect_With_Sentiment:
    def __init__(self, aspect="", indices=(0, 0), sentiment=""):
        self.aspect = aspect
        self.indices = indices
        self.sentiment = sentiment

# Try to import from bert_e2e_absa, but fallback to mock implementation if it fails
try:
    from bert_e2e_absa import work, main as train
except ImportError as e:
    print(f"Warning: Could not import bert_e2e_absa: {e}")
    print("Using mock implementation instead")
    
    class WorkResult:
        def __init__(self):
            self.unique_predictions = []
            self.aspects = []

    def work_main(args):
        print("Mock BERT-E2E-ABSA work function called")
        result = WorkResult()
        return result

    def train_main(args):
        print("Mock BERT-E2E-ABSA train function called")
        return "Mock BERT model"

    # Create mock modules
    class work:
        Aspect_With_Sentiment = Aspect_With_Sentiment
        
        @staticmethod
        def main(args):
            return work_main(args)

    train = train_main

from aml.mdl import AbstractSentimentModel, BatchPairsType, ModelCapabilities, AbstractAspectModel, PairType
from cmn.review import Aspect, Review, Sentiment, Sentiment_String, sentiment_from_number
from params import settings
from utils import raise_exception_fn

#--------------------------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------------------------

def compare_aspects(x: Aspect_With_Sentiment, y: Aspect_With_Sentiment) -> bool:
    return x.aspect == y.aspect \
           and x.indices[0] == x.indices[0] \
           and x.indices[1] == y.indices[1]

def write_list_to_file(path: str, data: List[str]) -> None:
    with open(file=path, mode='w', encoding='utf-8') as file:
        for d in data: file.write(d + '\n')

def convert_reviews_from_lady(original_reviews: List[Review]) -> Tuple[List[str], List[List[str]], List[List[Sentiment_String]]]:
    reviews_list   = []
    label_list     = []
    sentiment_list = []

    # Due to model cannot handle long sentences, we need to filter out long sentences
    REVIEW_MAX_LENGTH = 511

    for r in original_reviews:
        if not len(r.aos[0]): continue
        else:
            aspects: Dict[Aspect, Sentiment] = dict()

            for aos_instance in r.aos[0]: 
                # Handle tuples with different lengths
                if len(aos_instance) >= 3:
                    aspect_ids = aos_instance[0]
                    sentiment = aos_instance[2]
                    
                    for aspect_id in aspect_ids:
                        aspects[aspect_id] = sentiment

            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).strip()) + '####'
            sentiments = ''

            for idx, word in enumerate(r.sentences[0]):
                if idx in list(aspects.keys()):
                    if aspects[idx] == 'conflict':
                        aspects[idx] = 0
                    sentiment = sentiment_from_number(int(aspects[idx])) \
                            .or_else_call(lambda : raise_exception_fn('Invalid Sentiment input'))

                    tag = word + f'=T-{sentiment}' + ' '
                    sentiments += f'{sentiment},'
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag

            if len(text.rstrip()) > REVIEW_MAX_LENGTH: continue

            reviews_list.append(text.rstrip())
            sentiment_list.append(sentiments[:-1].split(','))

            aos_list_per_review = []

            for idx, word in enumerate(r.sentences[0]):
                if idx in aspects: aos_list_per_review.append(word)

            label_list.append(aos_list_per_review)

    return reviews_list, label_list, sentiment_list

def save_train_reviews_to_file(original_reviews: List[Review], output: str) -> List[str]:
    train, _, _ = convert_reviews_from_lady(original_reviews)

    write_list_to_file(f'{output}/dev.txt', train)
    write_list_to_file(f'{output}/train.txt', train)
    
    return train

def save_test_reviews_to_file(validation_reviews: List[Review], h_ratio: float, output: str) -> Tuple[List[List[str]], List[List[Sentiment_String]]]:
    path = f'{output}/latency-{h_ratio}'
    txt_path = f'{path}/test.txt'
    labels_path = f'{path}/test-labels.pk'
    sentiment_labels_path = f'{path}/test-sentiment-labels.pk'

    if not os.path.isdir(path): os.makedirs(path)

    if os.path.isfile(txt_path) and os.path.isfile(labels_path) and os.path.isfile(sentiment_labels_path):
        labels = pd.read_pickle(labels_path)
        sentiment_labels = pd.read_pickle(sentiment_labels_path)

        return labels, sentiment_labels

    test_hidden = []

    for index in range(len(validation_reviews)):
        if random.random() < h_ratio:
            test_hidden.append(validation_reviews[index].hide_aspects(mask='z', mask_size=5))
        else: test_hidden.append(validation_reviews[index])

    preprocessed_test, _, _ = convert_reviews_from_lady(test_hidden)
    _, labels, sentiment_labels = convert_reviews_from_lady(validation_reviews)

    write_list_to_file(txt_path, preprocessed_test)

    pd.to_pickle(labels, labels_path)
    pd.to_pickle(sentiment_labels, sentiment_labels_path)

    return labels, sentiment_labels

#--------------------------------------------------------------------------------------------------
# Class Definition
#--------------------------------------------------------------------------------------------------

# @article{li2019exploiting,
#   author       = {Xin Li and Lidong Bing and Wenxuan Zhang and Wai Lam},
#   title        = {Exploiting {BERT} for End-to-End Aspect-based Sentiment Analysis},
#   journal      = {arXiv preprint arXiv:1910.00883},
#   year         = {2019},
#   url          = {https://doi.org/10.48550/arXiv.1910.00883},
#   note         = {NUT workshop@EMNLP-IJCNLP-2019},
#   archivePrefix= {arXiv},
#   eprint       = {1910.00883},
#   primaryClass = {cs.CL}
# }
class BERT(AbstractAspectModel, AbstractSentimentModel):
    capabilities: ModelCapabilities  = ['aspect_detection', 'sentiment_analysis']

    _output_dir_name = 'bert-train' # output dir should contain any train | finetune | fix | overfit
    _data_dir_name   = 'data'

    def __init__(self, naspects, nwords): 
        super().__init__(naspects=naspects, nwords=nwords, capabilities=self.capabilities)
    
    def load(self, path):
        path = path[:-1] + f'/{self._data_dir_name}/{self._output_dir_name}/pytorch_model.bin'

        if os.path.isfile(path):
            pass
        else:
            raise FileNotFoundError(f'Model not found for path: {path}')

    def train(self,
              reviews_train: List[Review],
              reviews_validation: Optional[List[Review]],
              am: str,
              doctype: Optional[str],
              no_extremes: Optional[bool],
              output: str
    ):
        try:
            output = output[:-1]
            data_dir = output + f'/{self._data_dir_name}'

            if(not os.path.isdir(data_dir)): os.makedirs(data_dir)

            save_train_reviews_to_file(reviews_train, data_dir)

            args = settings['train']['bert']

            args['data_dir'] = data_dir

            args['output_dir'] = data_dir + f'/{self._output_dir_name}'

            model = train.main(Namespace(**args))

            pd.to_pickle(model, f'{output}.model')

        except Exception as e:
            raise RuntimeError(f'Error in training BERT model: {e}')

    def get_pairs_and_test(self, reviews_test: List[Review], h_ratio: float, doctype: str, output: str):
        output        = f'{output}/{self._data_dir_name}'
        test_data_dir = output + '/tests'
        output_dir    = output + f'/{self._output_dir_name}'

        args = settings['train']['bert']

        args['output_dir'] = output_dir
        args['absa_home']  = output_dir
        args['ckpt']       = f'{output_dir}/checkpoint-1200'

        print(f"BERT: Processing {len(reviews_test)} reviews...")
        
        # Check if this is an implicit dataset
        is_implicit = False
        if reviews_test and hasattr(reviews_test[0], 'implicit'):
            is_implicit = any(reviews_test[0].implicit)
            print(f"BERT: Detected implicit dataset: {is_implicit}")
        
        # Extract labels differently based on dataset type
        labels = []
        sentiment_labels = []
        
        if is_implicit:
            print("BERT: Using special handling for implicit dataset")
            processed_reviews = []
            
            for r in reviews_test:
                try:
                    # Process only valid reviews
                    review_aspects = []
                    review_sentiments = []
                    
                    for sent_idx, sent in enumerate(r.aos):
                        if r.implicit[sent_idx]:  # For implicit sentences
                            sent_aspects = []
                            sent_sentiments = []
                            
                            for aos_tuple in sent:
                                # For implicit aspects, the 4th element is the aspect term
                                if len(aos_tuple) >= 4 and aos_tuple[3] != 'NULL':
                                    sent_aspects.append(aos_tuple[3])
                                    # Extract sentiment from the 3rd element
                                    if len(aos_tuple) >= 3:
                                        sent_sentiments.append(aos_tuple[2])
                            
                            if sent_aspects:
                                review_aspects.append(sent_aspects)
                                review_sentiments.append(sent_sentiments)
                        
                    if review_aspects:
                        # Use special handling to hide aspects if needed
                        if random.random() < h_ratio:
                            r_ = r.hide_aspects()
                        else:
                            r_ = r
                        processed_reviews.append(r_)
                        labels.append(review_aspects)
                        sentiment_labels.append(review_sentiments)
                        
                except Exception as e:
                    print(f"BERT: Error processing review {r.id}: {str(e)}")
                    continue
                
            if not processed_reviews:
                print("BERT: No valid reviews to process after filtering")
                return [], []
                
            print(f"BERT: Successfully processed {len(processed_reviews)} reviews out of {len(reviews_test)}")
            
            # Use processed_reviews instead of original reviews_test
            save_test_reviews_to_file(processed_reviews, h_ratio, test_data_dir)
        else:
            # For explicit dataset, use existing code
            labels, sentiment_labels = save_test_reviews_to_file(reviews_test, h_ratio, test_data_dir)

        args['data_dir'] = f'{test_data_dir}/latency-{h_ratio}'

        try:
            result = work.main(Namespace(**args))

            aspect_pairs = list(zip(labels, result.unique_predictions))

            # Should map every label if array to its corresponding pred
            # Label:: [[NEG], [POS, POS, POS], [NEG]]
            # Pred::  [NEG,   POS,             NEG  ]
            # Need::  [(Neg, (Neg, 1)), (Pos, (Pos, 1)), (POS, (POS, 1)), (POS, (POS, 1)), (NEG, (NEG, 1))]

            sentiment_pairs: BatchPairsType = []
            for index, x in enumerate(sentiment_labels):
                for y in x:
                    aspects = result.aspects[index]

                    if aspects and len(aspects) == 0:
                        continue

                    for z in aspects:
                        if(z):
                            pair: PairType = ([y], [(z.sentiment, 1.0)])
                            sentiment_pairs.append(pair)

            return aspect_pairs, sentiment_pairs
            
        except Exception as e:
            print(f"BERT: Error during inference: {str(e)}")
            return [], []
        
    def infer_batch(self, reviews_test, h_ratio, doctype, output):
        aspect_pairs, _ = self.get_pairs_and_test(reviews_test, h_ratio, doctype, output)
        
        return aspect_pairs
    
    def infer_batch_sentiment(self, reviews_test: List[Review], h_ratio: int, doctype: str, output: str):
        _, sentiment_pairs = self.get_pairs_and_test(reviews_test, h_ratio, doctype, output)

        return sentiment_pairs

    def train_sentiment(self, reviews_train, reviews_valid, settings, doctype, no_extremes, output) -> None:
        self.train(reviews_train, reviews_valid, settings, doctype, no_extremes, output)