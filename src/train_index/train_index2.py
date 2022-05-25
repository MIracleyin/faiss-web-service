import numpy as np
import logging
import faiss
import pickle
import sys
from time import time
from config import DIM, INDEX_KEY, USE_GPU, index_path
from config import ch_para
from utils import get_database,chunk

sys.path.append("..")

index = faiss.index_factory(DIM, INDEX_KEY)
if USE_GPU:
    print("Use GPU ...")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

db = get_database(ch_para)
res = db.execute("select count(1) from scholar_embedded.demo_embeddings")
id_feat_iter = db.execute_iter(
    """
    SELECT ID, SPECTER
    FROM scholar_embedded.demo_embeddings
    """
)

t = time()
for idx, row in enumerate(chunk(id_feat_iter, 100*40)):
    tt = time()
    print(idx)
    id_feat_dict = {r[0]: r[1] for r in row}
    chunk_feat = np.array([])
    id_list, feat_list = [], []
    for id, feat in id_feat_dict.items():
        id_list.append(np.array(id))
        feat_list.append(np.array(feat, dtype=np.float32))
    id_array = np.array(id_list)
    feat_array = np.vstack(feat_list)
    index.add_with_ids(feat_array, id_array)
    index.train(feat_array)
    print('each', time()-tt)
    if idx == 10:
        break

print('10000', time() - t)
# save index
faiss.write_index(index, index_path)
