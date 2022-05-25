DEBUG = True

ch_para = {
    'host': "127.0.0.1",
    'user': "zvns342v4r6xhb",
    'password': "jvnh3cetktr78v",
    'port': 9000,
}

def GET_FAISS_RESOURCES():
    return None

def GET_FAISS_INDEX():
    raise NotImplementedError

def GET_FAISS_ID_TO_VECTOR():
    raise NotImplementedError

UPDATE_FAISS_AFTER_SECONDS = None

# Train
INDEX_KEY = "IDMap,Flat"
DIM = 768
# INDEX_KEY = "IDMap,IMI2x10,Flat"
# INDEX_KEY = "IDMap,OPQ16_64,IMI2x12,PQ8+16"
USE_GPU = False

index_path = "/home/yin/Public/ForkSource/faiss-web-service/resources1/index"
ids_vectors_path = '/home/yin/Public/ForkSource/faiss-web-service/resources1/ids_paths_vectors'

# Search
TOP_N = 5
SIMILARITY = 5