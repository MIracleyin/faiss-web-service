import numpy as np
import logging
import faiss
import pickle
import sys
from time import time
from config import DIM, INDEX_KEY, USE_GPU
from config import ch_para
from utils import get_database

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
index = faiss.index_factory(DIM, INDEX_KEY)
for idx, row in enumerate(id_feat_iter):
    print(idx)
    id, feat = row
    print(id)
    index.train(feat)
    index.add_with_ids(feat, id)
    if idx == 10000:
        break


print(time() - t)
