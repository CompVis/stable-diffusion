import os, sys
import numpy as np
import scann
import argparse
import glob
from multiprocessing import cpu_count
from tqdm import tqdm

from ldm.util import parallel_data_prefetch


def search_bruteforce(searcher):
    return searcher.score_brute_force().build()


def search_partioned_ah(searcher, dims_per_block, aiq_threshold, reorder_k,
                        partioning_trainsize, num_leaves, num_leaves_to_search):
    return searcher.tree(num_leaves=num_leaves,
                         num_leaves_to_search=num_leaves_to_search,
                         training_sample_size=partioning_trainsize). \
        score_ah(dims_per_block, anisotropic_quantization_threshold=aiq_threshold).reorder(reorder_k).build()


def search_ah(searcher, dims_per_block, aiq_threshold, reorder_k):
    return searcher.score_ah(dims_per_block, anisotropic_quantization_threshold=aiq_threshold).reorder(
        reorder_k).build()

def load_datapool(dpath):


    def load_single_file(saved_embeddings):
        compressed = np.load(saved_embeddings)
        database = {key: compressed[key] for key in compressed.files}
        return database

    def load_multi_files(data_archive):
        database = {key: [] for key in data_archive[0].files}
        for d in tqdm(data_archive, desc=f'Loading datapool from {len(data_archive)} individual files.'):
            for key in d.files:
                database[key].append(d[key])

        return database

    print(f'Load saved patch embedding from "{dpath}"')
    file_content = glob.glob(os.path.join(dpath, '*.npz'))

    if len(file_content) == 1:
        data_pool = load_single_file(file_content[0])
    elif len(file_content) > 1:
        data = [np.load(f) for f in file_content]
        prefetched_data = parallel_data_prefetch(load_multi_files, data,
                                                 n_proc=min(len(data), cpu_count()), target_data_type='dict')

        data_pool = {key: np.concatenate([od[key] for od in prefetched_data], axis=1)[0] for key in prefetched_data[0].keys()}
    else:
        raise ValueError(f'No npz-files in specified path "{dpath}" is this directory existing?')

    print(f'Finished loading of retrieval database of length {data_pool["embedding"].shape[0]}.')
    return data_pool


def train_searcher(opt,
                   metric='dot_product',
                   partioning_trainsize=None,
                   reorder_k=None,
                   # todo tune
                   aiq_thld=0.2,
                   dims_per_block=2,
                   num_leaves=None,
                   num_leaves_to_search=None,):

    data_pool = load_datapool(opt.database)
    k = opt.knn

    if not reorder_k:
        reorder_k = 2 * k

    # normalize
    # embeddings =
    searcher = scann.scann_ops_pybind.builder(data_pool['embedding'] / np.linalg.norm(data_pool['embedding'], axis=1)[:, np.newaxis], k, metric)
    pool_size = data_pool['embedding'].shape[0]

    print(*(['#'] * 100))
    print('Initializing scaNN searcher with the following values:')
    print(f'k: {k}')
    print(f'metric: {metric}')
    print(f'reorder_k: {reorder_k}')
    print(f'anisotropic_quantization_threshold: {aiq_thld}')
    print(f'dims_per_block: {dims_per_block}')
    print(*(['#'] * 100))
    print('Start training searcher....')
    print(f'N samples in pool is {pool_size}')

    # this reflects the recommended design choices proposed at
    # https://github.com/google-research/google-research/blob/aca5f2e44e301af172590bb8e65711f0c9ee0cfd/scann/docs/algorithms.md
    if pool_size < 2e4:
        print('Using brute force search.')
        searcher = search_bruteforce(searcher)
    elif 2e4 <= pool_size and pool_size < 1e5:
        print('Using asymmetric hashing search and reordering.')
        searcher = search_ah(searcher, dims_per_block, aiq_thld, reorder_k)
    else:
        print('Using using partioning, asymmetric hashing search and reordering.')

        if not partioning_trainsize:
            partioning_trainsize = data_pool['embedding'].shape[0] // 10
        if not num_leaves:
            num_leaves = int(np.sqrt(pool_size))

        if not num_leaves_to_search:
            num_leaves_to_search = max(num_leaves // 20, 1)

        print('Partitioning params:')
        print(f'num_leaves: {num_leaves}')
        print(f'num_leaves_to_search: {num_leaves_to_search}')
        # self.searcher = self.search_ah(searcher, dims_per_block, aiq_thld, reorder_k)
        searcher = search_partioned_ah(searcher, dims_per_block, aiq_thld, reorder_k,
                                                 partioning_trainsize, num_leaves, num_leaves_to_search)

    print('Finish training searcher')
    searcher_savedir = opt.target_path
    os.makedirs(searcher_savedir, exist_ok=True)
    searcher.serialize(searcher_savedir)
    print(f'Saved trained searcher under "{searcher_savedir}"')

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--database',
                        '-d',
                        default='data/rdm/retrieval_databases/openimages',
                        type=str,
                        help='path to folder containing the clip feature of the database')
    parser.add_argument('--target_path',
                        '-t',
                        default='data/rdm/searchers/openimages',
                        type=str,
                        help='path to the target folder where the searcher shall be stored.')
    parser.add_argument('--knn',
                        '-k',
                        default=20,
                        type=int,
                        help='number of nearest neighbors, for which the searcher shall be optimized')

    opt, _  = parser.parse_known_args()

    train_searcher(opt,)