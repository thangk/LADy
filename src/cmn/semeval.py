import os
from tqdm import tqdm
import xml.etree.ElementTree as et
import logging
import sys

#nlp = spacy.load("en_core_web_sm")  # en_core_web_trf for transformer-based; error ==> python -m spacy download en_core_web_sm

from cmn.review import Review

# Set up logging
debug_log = logging.getLogger('semeval_debug')
debug_log.setLevel(logging.DEBUG)
# Add a file handler
os.makedirs('logs', exist_ok=True)
fh = logging.FileHandler('logs/semeval_debug.log', mode='w')
fh.setLevel(logging.DEBUG)
# Add formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
debug_log.addHandler(fh)

class SemEvalReview(Review):

    def __init__(self, id, sentences, time, author, aos): super().__init__(self, id, sentences, time, author, aos)

    @staticmethod
    def load(path, explicit=True, implicit=False):
        debug_log.info(f"SemEvalReview.load called with path={path}, explicit={explicit}, implicit={implicit}")
        # Auto-detect if this is an implicit dataset based on filename
        if 'implicit' in path.lower() and not implicit:
            debug_log.info("Detected implicit dataset from filename, setting implicit=True")
            implicit = True
            
        if str(path).endswith('.xml'): return SemEvalReview._xmlloader(path, explicit, implicit)
        return SemEvalReview._txtloader(input)

    @staticmethod
    def _txtloader(path):
        reviews = []
        with tqdm(total=os.path.getsize(path)) as pbar, open(path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                pbar.update(len(line))
                sentence, aos = line.split('####')
                aos = aos.replace('\'POS\'', '+1').replace('\'NEG\'', '-1').replace('\'NEU\'', '0')

                # for the current datafile, each row is a review of single sentence!
                # sentence = nlp(sentence)
                reviews.append(Review(id=i, sentences=[[str(t).lower() for t in sentence.split()]], time=None, author=None,
                                      aos=[eval(aos)], lempos=None,
                                      parent=None, lang='eng_Latn'))
        return reviews

    @staticmethod
    def _xmlloader(path, explicit, implicit):
        reviews_list = []
        debug_log.info(f"Loading XML file from: {path}")
        debug_log.info(f"explicit={explicit}, implicit={implicit}")
        xtree = et.parse(path).getroot()
        debug_log.info(f"XML root tag: {xtree.tag}")
        debug_log.info(f"Number of children: {len(xtree)}")
        
        if xtree.tag == 'Reviews':   
            debug_log.info("Processing 'Reviews' format")
            reviews = [SemEvalReview._parse(xsentence, explicit, implicit) for xreview in tqdm(xtree) for xsentences in xreview for xsentence in xsentences]
        if xtree.tag == 'sentences': 
            debug_log.info("Processing 'sentences' format")
            reviews = [SemEvalReview._parse(xsentence, explicit, implicit) for xsentence in tqdm(xtree)]

        result = [r for r in reviews if r]
        debug_log.info(f"Loaded {len(result)} reviews with aspects")
        return result

    @staticmethod
    def _map_idx(aspect, text):
        # aspect: ('token', from_char, to_char)
        text_tokens = text[:aspect[1]].split()
        # to fix if  "aaaa ,b, c" ",b c" if b is the aspect
        if len(text_tokens) > 0 and not text[aspect[1] - 1].isspace(): text_tokens.pop()
        aspect_tokens = aspect[0].split()

        # tmp = [*text] #mutable string :)
        # # these two blank space add bug to the char indexes for aspects if a sentence have multiple aspects!
        # tmp[aspect[1]: aspect[2]] = [' '] + [*aspect[0]] + [' ']
        # text = ''.join(tmp)

        return [i for i in range(len(text_tokens), len(text_tokens) + len(aspect_tokens))]

    @staticmethod
    def _parse(xsentence, explicit, implicit):
        id = xsentence.attrib["id"]
        debug_log.info(f"Parsing sentence {id}, explicit={explicit}, implicit={implicit}")
        aos = []; aos_cats = []
        for element in xsentence:
            if element.tag == 'text': 
                sentence = element.text # we consider each sentence as a signle review
                debug_log.info(f"Found text: {sentence[:30]}...")
            elif element.tag == 'Opinions':#semeval-15-16
                debug_log.info("Found Opinions tag")
                #<Opinion target="place" category="RESTAURANT#GENERAL" polarity="positive" from="5" to="10"/>
                for opinion in element:
                    # Load implicit, explicit, or both aspects
                    if not implicit and opinion.attrib["target"] == 'NULL': continue
                    if not explicit and opinion.attrib["target"] != 'NULL': continue
                    # we may have duplicates for the same aspect due to being in different category like in semeval 2016's <sentence id="1064477:4">
                    aspect = (opinion.attrib["target"], int(opinion.attrib["from"]), int(opinion.attrib["to"])) #('place', 5, 10)
                    # we need to map char index to token index in aspect
                    aspect = SemEvalReview._map_idx(aspect, sentence)
                    category = opinion.attrib["category"] # 'RESTAURANT#GENERAL'
                    sentiment = opinion.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0') #'+1'
                    aos.append((aspect, [], sentiment, opinion.attrib["target"]))
                    aos_cats.append(category)
                aos = sorted(aos, key=lambda x: int(x[0][0])) #based on start of sentence

            elif element.tag == 'aspectTerms':#semeval-14
                debug_log.info("Found aspectTerms tag")
                #<aspectTerm term="table" polarity="neutral" from="5" to="10"/>
                for opinion in element:
                    # Load implicit, explicit, or both aspects
                    if not implicit and opinion.attrib["term"] == 'NULL': continue
                    if not explicit and opinion.attrib["term"] != 'NULL': continue
                    # we may have duplicates for the same aspect due to being in different category like in semeval 2016's <sentence id="1064477:4">
                    aspect = (opinion.attrib["term"], int(opinion.attrib["from"]), int(opinion.attrib["to"])) #('place', 5, 10)
                    # we need to map char index to token index in aspect
                    aspect = SemEvalReview._map_idx(aspect, sentence)
                    sentiment = opinion.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0') #'+1'
                    aos.append((aspect, [], sentiment, opinion.attrib["term"]))

                aos = sorted(aos, key=lambda x: int(x[0][0])) #based on start of sentence

            elif element.tag == 'aspectCategories':  # semeval-14
                debug_log.info(f"Found aspectCategories tag with {len(element)} categories")
                for i, opinion in enumerate(element):
                    #<aspectCategory category="food" polarity="neutral"/>
                    category = opinion.attrib["category"]
                    sentiment = opinion.attrib["polarity"].replace('positive', '+1').replace('negative', '-1').replace('neutral', '0')
                    debug_log.info(f"  - Category: {category}, Sentiment: {sentiment}")
                    aos_cats.append(category)

                    # For implicit datasets, create implicit aspects
                    if implicit:
                        # Use dummy token indices (will be marked as implicit below)
                        dummy_aspect = ([i], [], sentiment, 'NULL')  # Use i for index to make them unique
                        aos.append(dummy_aspect)
                        debug_log.info(f"    Added implicit aspect for category {category}")

        # Mark all aos with implicit aspects
        implicit_arr = [False] * len(aos)
        if implicit:
            for i, (idxlist, o, s, aspect_token) in enumerate(aos):
                if aspect_token == 'NULL': implicit_arr[i] = True

        if 'sentence' not in locals():
            debug_log.error(f"No text element found in sentence {id}")
            return None
            
        #sentence = nlp(sentence) # as it does some processing, it destroys the token idx for aspect term
        tokens = sentence.split()
        # to fix ",a b c," to "a b c"
        # to fix '"sales" team' to 'sales team' => semeval-14-labptop-<sentence id="1316">
        # todo: fix 'Food-awesome.' to 'food awesome' => semeval-14-restaurant-<sentence id="1817">
        for i, (idxlist, o, s, aspect_token) in enumerate(aos):
            for j, idx in enumerate(idxlist):
                if not implicit_arr[i]:
                    tokens[idx] = aspect_token.split()[j].replace('"', '')
                # Preserve the 4th element (aspect_token) when updating aos
                aos[i] = (idxlist, o, s, aspect_token)

        debug_log.info(f"Finished parsing sentence {id} with {len(aos)} aspects")
        return Review(id=id, sentences=[[str(t).lower() for t in tokens]], time=None, author=None,
                      aos=[aos], lempos=None,
                      parent=None, lang='eng_Latn', category=aos_cats, implicit=implicit_arr) if aos else None

