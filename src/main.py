from typing import List, Tuple, Set, Union
import argparse, os, json, time, re
from tqdm import tqdm
import numpy as np, pandas as pd
import pampy

import pytrec_eval
from nltk.corpus import wordnet as wn

import params
from cmn.review import Review
from aml.mdl import AbstractAspectModel, AbstractSentimentModel, ModelCapabilities, ModelCapability

# ---------------------------------------------------------------------------------------
# Typings
# ---------------------------------------------------------------------------------------
Aspects = List[List[str]]

PredictedAspect = List[Tuple[int, float]]  # Tuple containing index and weight

PairType = Tuple[Aspects, PredictedAspect]

# ---------------------------------------------------------------------------------------
# Logics
# ---------------------------------------------------------------------------------------
def load(input, output, cache=True):
    print('\n1. Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        if not cache: raise FileNotFoundError
        print(f'1.1. Loading existing processed reviews file {output}...')
        reviews = pd.read_pickle(output)
        print(f'Loaded {len(reviews)} reviews from pickle file')
        return reviews

    except (FileNotFoundError, EOFError) as _:
        try:
            print('1.1. Loading existing processed pickle file failed! Loading raw reviews ...')
            print(f'Input path: {input}')
            if "semeval" in input.lower():
                print(f'Detected semeval dataset in path: {input}')
                from cmn.semeval import SemEvalReview
                implicit = "implicit" in input.lower()
                print(f'Setting implicit={implicit} based on input path')
                reviews = SemEvalReview.load(input, explicit=not implicit, implicit=implicit)
                print(f'After SemEvalReview.load, got {len(reviews)} reviews')
            elif "twitter" in input.lower():
                from cmn.twitter import TwitterReview
                reviews = TwitterReview.load(input)
            else:
                # from cmn.mams import MAMSReview
                print("No specific dataset ('semeval' or 'twitter') was detected in the input.")
                print(f"Input was: {input}")
                return []  # Return empty list to avoid crashing later
                
            print(f'(#reviews: {len(reviews)})')
            print(f'\n1.2. Augmentation via backtranslation by {params.settings["prep"]["langaug"]} {"in batches" if params.settings["prep"] else ""}...')
            for lang in params.settings['prep']['langaug']:
                if lang:
                    print(f'\n{lang} ...')
                    if params.settings['prep']['batch']:
                        start = time.time()
                        Review.translate_batch(reviews, lang, params.settings['prep']) #all at once, esp., when using gpu
                        end = time.time()
                        print(f'{lang} done all at once (batch). Time: {end - start}')
                    else:
                        for r in tqdm(reviews): r.translate(lang, params.settings['prep'])

                # to save a file per language. I know, it has a minor logical bug as the save file include more languages!
                output_ = output
                for l in params.settings['prep']['langaug']:
                    if l and l != lang:
                        output_ = output_.replace(f'{l}.', '')
                pd.to_pickle(reviews, output_)

            print(f'\n1.3. Saving processed pickle file {output}...')
            pd.to_pickle(reviews, output)
            return reviews
        except Exception as error:
            print(f'Error...{error}')
            raise error

def split(nsample, output):
    # We split originals into train, valid, test. So each have its own augmented versions.
    # During test (or even train), we can decide to consider augmented version or not.

    from sklearn.model_selection import KFold, train_test_split
    from json import JSONEncoder

    train, test = train_test_split(np.arange(nsample), train_size=params.settings['train']['ratio'], random_state=params.seed, shuffle=True)

    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    if params.settings['train']['nfolds'] == 0:
        splits['folds'][0] = dict()
        splits['folds'][0]['train'] = train
        splits['folds'][0]['valid'] = []
    elif params.settings['train']['nfolds'] == 1:
        splits['folds'][0] = dict()
        splits['folds'][0]['train'] = train[:len(train)//2]
        splits['folds'][0]['valid'] = train[len(train)//2:]
    else:
        skf = KFold(n_splits=params.settings['train']['nfolds'], random_state=params.seed, shuffle=True)
        for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
            splits['folds'][k] = dict()
            splits['folds'][k]['train'] = train[trainIdx]
            splits['folds'][k]['valid'] = train[validIdx]

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            return JSONEncoder.default(self, obj)

    with open(f'{output}/splits.json', 'w') as f: json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)
    return splits

def train(args, am, train, valid, f, output, capability: ModelCapability):
    print(f'\n2. Aspect model training for {am.name} ...')
    print('#' * 50)
    try:
        print(f'2.1. Loading saved aspect model from {output}/f{f}. ...')
        am.load(f'{output}/f{f}.')
    except (FileNotFoundError, EOFError) as _:
        print(f'2.1. Loading saved aspect model failed! Training {am.name} for {args.naspects} of aspects. See {output}/f{f}.model.train.log for training logs ...')
        if not os.path.isdir(output): os.makedirs(output)
        get_model_train_method(am, capability)(train, valid, params.settings['train'][am.name()], params.settings['prep']['doctype'], params.settings['train']['no_extremes'], f'{output}/f{f}.')

        # from aml.mdl import AbstractAspectModel
        print('2.2. Quality of aspects ...')
        for q in params.settings['train']['qualities']: print(f'({q}: {am.quality(q)})')

def test(am, test, f, output: str, capability: ModelCapability):
    cp_name = get_capability_short_name(capability)
    model_job = capability.split("_")[0].capitalize()

    print(f'\n3. {model_job} model testing for {am.name} ...')
    print('#' * 50)
    try:
        print(f'\n3.1. Loading saved predictions on test set from {output}f{f}.{cp_name}.model.pred.{params.settings["test"]["h_ratio"]} ...')
        return pd.read_pickle(f'{output}f{f}.model.{cp_name}.pred.{params.settings["test"]["h_ratio"]}')
    except (FileNotFoundError, EOFError) as _:
        print(f'\n3.1. Loading saved predictions on test set failed! Predicting on the test set with {params.settings["test"]["h_ratio"] * 100}% latent aspect ...')
        print(f'3.2. Loading {model_job} model from {output}f{f}.{cp_name}.model for testing ...')
        am.load(f'{output}/f{f}.')
        print(f'3.3. Testing {model_job} model ...')
        pairs = get_model_infer_method(am, capability)(reviews_test=test, h_ratio=params.settings['test']['h_ratio'], doctype=params.settings['prep']['doctype'], output=f'{output}/f{f}')
        pd.to_pickle(pairs, f'{output}f{f}.model.{cp_name}.pred.{params.settings["test"]["h_ratio"]}')

def get_model_infer_method(am: Union[AbstractSentimentModel, AbstractAspectModel], model_capability: ModelCapability):
    if isinstance(am, AbstractAspectModel) and model_capability == 'aspect_detection':
        return am.infer_batch
    elif isinstance(am, AbstractSentimentModel) and model_capability == 'sentiment_analysis':
        return am.infer_batch_sentiment
    
    raise Exception(f'Not handled model: {am.name()}')

def get_model_train_method(am: Union[AbstractSentimentModel, AbstractAspectModel], model_capability: ModelCapability):
    if isinstance(am, AbstractAspectModel) and model_capability == 'aspect_detection':
        return am.train
    elif isinstance(am, AbstractSentimentModel) and model_capability == 'sentiment_analysis':
        return am.train_sentiment
    
    raise Exception(f'Not handled model: {am.name()}')


def get_model_metrics(model_capability: ModelCapability) -> Set[str]:
    return pampy.match(model_capability, 
        'aspect_detection', set(f'{m}_{",".join([str(i) for i in params.settings["eval"]["aspect_detection"]["topkstr"]])}' for m in params.settings['eval']['aspect_detection']['metrics']),
        'sentiment_analysis', set(f'{m}_{",".join([str(i) for i in params.settings["eval"]["sentiment_analysis"]["topkstr"]])}' for m in params.settings['eval']['sentiment_analysis']['metrics']),
    ) #type: ignore

def get_capability_short_name(cp: ModelCapability) -> str:
    return pampy.match(cp,
        'sentiment_analysis', 'sa',
        'aspect_detection', 'ad',
    ) # type: ignore

def evaluate(input: str, output: str, model_capability: ModelCapability):
    model_job = model_capability.split('_')[0].capitalize()
    print(f'\n4. {model_job} model evaluation for {input} ...')
    print('#' * 50)
    pairs = pd.read_pickle(input)
    metrics_set = get_model_metrics(model_capability)

    # Debug the pairs data structure
    print(f"DEBUG: Loaded {len(pairs)} pairs from {input}")
    if pairs and len(pairs) > 0:
        print(f"DEBUG: First pair structure: {type(pairs[0])}")
        print(f"DEBUG: First ground truth (pair[0]): {pairs[0][0]}")
        print(f"DEBUG: First prediction (pair[1]): {pairs[0][1]}")
    
    qrel = dict()
    run = dict()

    print(f'\n4.1. Building pytrec_eval input for {len(pairs)} instances ...')
    for i, pair in enumerate(pairs):
        # Ensure pair[0] (ground truth aspects) is not empty before proceeding
        if not pair[0]:
            print(f"DEBUG: Skipping evaluation for instance {i} due to empty ground truth aspects.")
            continue
            
        if params.settings['eval']['syn']:
            syn_list = set()
            # Ensure p_instance is a string or handle appropriately
            for p_instance in pair[0]:
                if isinstance(p_instance, str):
                    syn_list.add(p_instance)
                    syn_list.update(set([lemma.name() for syn in wn.synsets(p_instance) for lemma in syn.lemmas()]))
                else:
                    # Handle cases where p_instance might not be a string (e.g., for rnd model)
                    print(f"DEBUG: Non-string aspect '{p_instance}' found in instance {i}, converting to string representation")
                    syn_list.add(str(p_instance))
            qrel['q' + str(i)] = {w: 1 for w in syn_list if isinstance(w, str)}
        else:
            # Ensure elements in pair[0] are suitable as dictionary keys (e.g., strings)
            # Convert lists or tuples to their string representation
            qrel['q' + str(i)] = {}
            for w in pair[0]:
                if isinstance(w, (list, tuple)):
                    key = str(w)  # Convert complex structure to string
                else:
                    key = str(w)  # Ensure string for dictionary key
                qrel['q' + str(i)][key] = 1

        # the prediction list may have duplicates
        run['q' + str(i)] = {}
        for j, (w, p) in enumerate(pair[1]):
            # Ensure predicted aspect 'w' is suitable as a dictionary key
            if isinstance(w, (list, tuple)):
                key = str(w)  # Convert complex structure to string
            else:
                key = str(w)  # Ensure string for dictionary key
                
            if key not in run['q' + str(i)].keys():
                run['q' + str(i)][key] = len(pair[1]) - j
    
    # Debug counts of qrel and run entries
    print(f"DEBUG: Total qrel entries: {len(qrel)}")
    if qrel:
        qrel_sizes = [len(qrel[q]) for q in qrel]
        print(f"DEBUG: Average items per qrel entry: {sum(qrel_sizes)/len(qrel_sizes) if qrel_sizes else 0}")
        print(f"DEBUG: Min items in a qrel entry: {min(qrel_sizes) if qrel_sizes else 0}")
        print(f"DEBUG: Max items in a qrel entry: {max(qrel_sizes) if qrel_sizes else 0}")
    
    print(f"DEBUG: Total run entries: {len(run)}")
    if run:
        run_sizes = [len(run[r]) for r in run]
        print(f"DEBUG: Average items per run entry: {sum(run_sizes)/len(run_sizes) if run_sizes else 0}")
        print(f"DEBUG: Min items in a run entry: {min(run_sizes) if run_sizes else 0}")
        print(f"DEBUG: Max items in a run entry: {max(run_sizes) if run_sizes else 0}")

    print(f'4.2. Calling pytrec_eval for {metrics_set} ...')
    # Filter out queries that were skipped (had empty qrel entries)
    valid_queries = list(qrel.keys())
    filtered_run = {qid: run[qid] for qid in valid_queries if qid in run}
    
    # -- DEBUGGING --
    print(f"DEBUG: len(qrel) = {len(qrel)}, len(filtered_run) = {len(filtered_run)}")
    if not qrel or not filtered_run:
        print("Error: No valid qrel or run data to evaluate. Skipping pytrec_eval.")
        # Create an empty CSV with headers to satisfy agg function
        with open(output, 'w') as f:
            metrics = set()
            for m in params.settings['eval'][model_capability]['metrics']:
                for k in params.settings['eval'][model_capability]['topkstr']:
                    metrics.add(f"{m}_{k}")
            
            f.write("metric,mean\n")
            for metric in metrics:
                f.write(f"{metric},0.0\n")
        
        print(f"Created empty CSV file with headers at {output}")
        return pd.DataFrame() # Return empty DataFrame
    # -- END DEBUGGING --
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics_set)
    results = evaluator.evaluate(filtered_run)

    # -- DEBUGGING --
    print(f"DEBUG: pytrec_eval results dict type: {type(results)}")
    if results and len(results) < 5: # Print first few results if not too long
        print(f"DEBUG: First few pytrec_eval results: { {k: results[k] for k in list(results)[:5]} }")
    elif not results:
        print("DEBUG: pytrec_eval results dict is empty!")
    # -- END DEBUGGING --
    
    df = pd.DataFrame.from_dict(results)

    # -- DEBUGGING --
    print(f"DEBUG: DataFrame df shape: {df.shape}")
    if not df.empty:
        print(f"DEBUG: DataFrame df head:\n{df.head()}")
    # -- END DEBUGGING --

    df_mean = df.mean(axis=1).to_frame('mean')

    # -- DEBUGGING --
    print(f"DEBUG: DataFrame df_mean shape: {df_mean.shape}")
    if not df_mean.empty:
        print(f"DEBUG: DataFrame df_mean:\n{df_mean}")
    # -- END DEBUGGING --
    
    # Explicitly handle empty df_mean before saving
    if df_mean.empty:
        print(f"WARNING: Mean DataFrame is empty. Creating an empty CSV file at {output}")
        # Create an empty file with metrics that match the agg() function expectations
        metrics = set()
        for m in params.settings['eval'][model_capability]['metrics']:
            for k in params.settings['eval'][model_capability]['topkstr']:
                metrics.add(f"{m}_{k}")
        
        # Create a DataFrame with the right structure
        empty_df = pd.DataFrame({
            'metric': list(metrics),
            'mean': [0.0] * len(metrics)
        })
        empty_df.to_csv(output, index=False)
        print(f"Created properly formatted empty CSV at {output}")
        return empty_df
    else:
        # Add 'metric' column for consistent structure
        df_mean = df_mean.reset_index().rename(columns={'index': 'metric'})
        print(f"Saving mean DataFrame to {output}")
        df_mean.to_csv(output, index=False)
        return df_mean

def agg(path, output):  # can be reused for many metric dataframes
    #TODO: add filter on path
    print(f'\n5. Aggregating results in {path} in {output} ...')

    files_found = []
    for dirpath, _, filenames in os.walk(path):
        files_found += [
            os.path.join(dirpath, file)
            for file in filenames
            if file.endswith('.csv')
        ]
    
    print(f"Found {len(files_found)} CSV files for aggregation:")
    for file in files_found:
        print(f"  - {file}")
    
    if not files_found:
        print(f"No CSV files found in {path}. Skipping aggregation.")
        return

    # Create dummy data if all CSV files are empty
    all_empty = True
    valid_files = []
    
    for file in files_found:
        try:
            df = pd.read_csv(file)
            if not df.empty and 'mean' in df.columns:
                all_empty = False
                valid_files.append(file)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if all_empty:
        print("All CSV files are empty or invalid. Creating dummy aggregate file.")
        # Create a dummy aggregate file with all zeros
        metrics = set()
        for file in files_found:
            try:
                with open(file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('metric'):
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 1:
                                metrics.add(parts[0])
            except:
                pass
        
        if not metrics:
            # If we couldn't read any metrics from files, use default ones
            metrics = {
                'P_1', 'P_5', 'P_10', 'P_100',
                'recall_1', 'recall_5', 'recall_10', 'recall_100',
                'ndcg_cut_1', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_100',
                'map_cut_1', 'map_cut_5', 'map_cut_10', 'map_cut_100',
                'success_1', 'success_5', 'success_10', 'success_100'
            }
        
        # Create dummy DataFrame
        data = {'metric': list(metrics), 'mean': [0.0] * len(metrics)}
        pd.DataFrame(data).to_csv(f'{output}/agg.ad.pred.eval.mean.csv', index=False)
        return
        
    # Normal aggregation with valid files
    all_results = pd.DataFrame()
    
    for filepath in valid_files:
        try:
            # Extract fold number from file name
            match = re.search(r'f(\d+)', os.path.basename(filepath))
            if match:
                foldi = match.group(1)
                fold_col = f'fold{foldi}'
            else:
                fold_col = f'unknown_fold_{len(all_results.columns)}'
            
            # Read CSV with correct formatting
            result = pd.read_csv(filepath)
            
            # If first file, use it as the base with the 'metric' column
            if all_results.empty:
                all_results = result.copy()
                # Rename the mean column to preserve it
                if 'mean' in all_results.columns:
                    all_results.rename(columns={'mean': 'mean_avg'}, inplace=True)
            else:
                # For subsequent files, only add the mean column with the fold name
                if 'mean' in result.columns and 'metric' in result.columns:
                    # Ensure metric columns align
                    result_dict = dict(zip(result['metric'], result['mean']))
                    
                    # Add the mean values to the correct rows by metric
                    all_results[fold_col] = all_results['metric'].map(result_dict)
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    # Make sure we have a mean column if it was renamed
    if 'mean_avg' in all_results.columns:
        all_results.rename(columns={'mean_avg': 'mean'}, inplace=True)
    
    # Calculate the mean across all folds
    if len(valid_files) > 1:
        fold_cols = [col for col in all_results.columns if col.startswith('fold')]
        if fold_cols:
            all_results['mean'] = all_results[fold_cols].mean(axis=1)
    
    # Ensure output directory exists
    os.makedirs(output, exist_ok=True)
    
    # Save with metric as index
    all_results.set_index('metric', inplace=True)
    all_results.to_csv(f'{output}/agg.ad.pred.eval.mean.csv')

def main(args):
    if not os.path.isdir(args.output): os.makedirs(args.output)
    
    debug_file = os.path.join(args.output, "debug.txt")
    with open(debug_file, "w") as f:
        f.write(f"Starting with args: {args}\n")
        f.write(f"Input file: {args.data}\n")

    langaug_str = '.'.join([l for l in params.settings['prep']['langaug'] if l])
    reviews_pkl = f'{args.output}/reviews.{langaug_str}.pkl'.replace('..pkl', '.pkl')

    with open(debug_file, "a") as f:
        f.write(f"Loading reviews from {args.data} to {reviews_pkl}\n")

    try:
        reviews = load(args.data, reviews_pkl)
        print(f"Loaded {len(reviews)} reviews successfully")
        
        with open(debug_file, "a") as f:
            f.write(f"Loaded {len(reviews)} reviews\n")
            
        # The regular flow continues from here
        try:
            print("Getting ready to split the data...")
            splits = split(len(reviews), args.output)
            print("Split successful")
            
            with open(debug_file, "a") as f:
                f.write("Split data into train/test sets\n")
                
            output = f'{args.output}/{args.naspects}.{langaug_str}'.rstrip('.')
            print(f"Output path set to {output}")
            
            # Set up model
            if 'rnd' == args.am: from aml.rnd import Rnd; am = Rnd(args.naspects, params.settings['train']['nwords']) 
            if 'lda' == args.am: from aml.lda import Lda; am = Lda(args.naspects, params.settings['train']['nwords'])
            if 'btm' == args.am: from aml.btm import Btm; am = Btm(args.naspects, params.settings['train']['nwords'])
            if 'ctm' == args.am: from aml.ctm import Ctm; am = Ctm(args.naspects, params.settings['train']['nwords'], params.settings['train']['ctm']['contextual_size'], params.settings['train']['ctm']['num_samples'])
            if 'bert' == args.am: from aml.bert import BERT; am = BERT(args.naspects, params.settings['train']['nwords'])
            print(f"Created model of type {args.am}")
            
            with open(debug_file, "a") as f:
                f.write(f"Created {args.am} model\n")
                
            if not os.path.isdir(output): os.makedirs(output)
            print(f"Created output directory: {output}")
            
            output_dir = f'{output}/{args.am}/'
            if not os.path.isdir(output_dir): os.makedirs(output_dir)
            print(f"Created model output directory: {output_dir}")
            
            with open(debug_file, "a") as f:
                f.write(f"Created output directories\n")
                
            # Train model
            model_capability = 'aspect_detection'
            print(f"Using model capability: {model_capability}")
            
            # Run model on each fold
            for fold_idx in splits['folds'].keys():
                print(f"Processing fold {fold_idx}...")
                train_idx = splits['folds'][fold_idx]['train']
                valid_idx = splits['folds'][fold_idx]['valid']
                print(f"Train set size: {len(train_idx)}, Valid set size: {len(valid_idx)}")
                
                with open(debug_file, "a") as f:
                    f.write(f"Training fold {fold_idx}: train={len(train_idx)}, valid={len(valid_idx)}\n")
                
                train(args=args, am=am, train=[reviews[i] for i in train_idx],
                      valid=[reviews[i] for i in valid_idx] if len(valid_idx) > 0 else [],
                      f=fold_idx, output=output_dir, capability=model_capability)
                print(f"Fold {fold_idx} trained successfully")
            
            with open(debug_file, "a") as f:
                f.write("All folds trained successfully\n")
            
            # Test on all test data
            test_results = {}
            with open(debug_file, "a") as f:
                f.write("Starting test loop...\n")
            for fold_idx in splits['folds'].keys():
                print(f"Testing fold {fold_idx}...")
                with open(debug_file, "a") as f:
                    f.write(f"Testing fold {fold_idx}...\n")
                test(am=am, test=[reviews[i] for i in splits['test']], f=fold_idx, output=output_dir, capability=model_capability)
                print(f"Fold {fold_idx} testing complete.")
                with open(debug_file, "a") as f:
                    f.write(f"Fold {fold_idx} testing complete.\n")
            
            with open(debug_file, "a") as f:
                f.write("Test loop finished.\n")

            # Evaluate on all test data
            with open(debug_file, "a") as f:
                f.write("Starting evaluate loop...\n")
            for fold_idx in splits['folds'].keys():
                print(f"Evaluating fold {fold_idx}...")
                with open(debug_file, "a") as f:
                    f.write(f"Evaluating fold {fold_idx}...\n")
                    
                input_pred_file = f'{output_dir}f{fold_idx}.model.ad.pred.{params.settings["test"]["h_ratio"]}'
                output_eval_file = f'{input_pred_file}.ad.eval.mean.csv' # Potential path issue here?
                
                with open(debug_file, "a") as f:
                    f.write(f"  Input pred file: {input_pred_file}\n")
                    f.write(f"  Output eval file: {output_eval_file}\n")
                    
                evaluate(input=input_pred_file, output=output_eval_file, model_capability=model_capability)
                print(f"Fold {fold_idx} evaluation complete.")
                with open(debug_file, "a") as f:
                    f.write(f"Fold {fold_idx} evaluation complete.\n")
            
            with open(debug_file, "a") as f:
                f.write("Evaluate loop finished.\n")
            
            # Add aggregation step within main() with debug
            if 'agg' in params.settings['cmd'] and not args.skip_agg:
                with open(debug_file, "a") as f:
                    f.write(f"Running aggregation on {output_dir} (results) and saving to {args.output} (output)\n")
                    
                # List CSV files to debug
                csv_files = []
                for dirpath, _, filenames in os.walk(output_dir):
                    csv_files += [
                        os.path.join(dirpath, file)
                        for file in filenames
                        if file.endswith('.csv')
                    ]
                with open(debug_file, "a") as f:
                    f.write(f"Found {len(csv_files)} CSV files for aggregation:\n")
                    for csv_file in csv_files:
                        f.write(f"  - {csv_file}\n")
                    
                try:
                    agg(output_dir, args.output)
                    with open(debug_file, "a") as f:
                        f.write(f"Aggregation completed successfully\n")
                except Exception as e:
                    with open(debug_file, "a") as f:
                        f.write(f"Error during aggregation: {str(e)}\n")
                        import traceback
                        f.write(traceback.format_exc())

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            with open(debug_file, "a") as f:
                f.write(f"Error during processing: {str(e)}\n")
                f.write(traceback.format_exc())
    except Exception as e:
        print(f"Error loading reviews: {str(e)}")
        import traceback
        traceback.print_exc()
        with open(debug_file, "a") as f:
            f.write(f"Error loading reviews: {str(e)}\n")
            f.write(traceback.format_exc())

# {CUDA_VISIBLE_DEVICES=0,1} won't work https://discuss.pytorch.org/t/using-torch-data-prallel-invalid-device-string/166233
# TOKENIZERS_PARALLELISM=true
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/SemEval-14/Laptop_Train_v2.xml -output ../output/SemEval-14/Laptop 2>&1 | tee ../output/SemEval-14/Laptop/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/toy.2016SB5 2>&1 | tee ../output/toy.2016SB5/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml -output ../output/SemEval-14/Restaurants 2>&1 | tee ../output/SemEval-14/Restaurants/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml -output ../output/2015SB12 2>&1 | tee ../output/semeval+/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml/log.txt &
# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python -u main.py -am lda -naspect 5 -data ../data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/2016SB5 2>&1 | tee ../output/2016SB5/log.txt &

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('-am', type=str.lower, default='rnd', help='aspect modeling method (eg. --am lda)')
    parser.add_argument('-data', dest='data', type=str, default='../data/raw/twitter/acl-14-short-data/toy.raw', help='raw dataset file path, e.g., -data ..data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml')
    parser.add_argument('-output', dest='output', type=str, default='../output/twitter-agg', help='output path, e.g., -output ../output/semeval/2016.xml')
    parser.add_argument('-naspects', dest='naspects', type=int, default=25, help='user-defined number of aspects, e.g., -naspect 25')
    parser.add_argument('-skip-agg', action='store_true', help='skip the aggregation step')
    args = parser.parse_args()

    main(args)