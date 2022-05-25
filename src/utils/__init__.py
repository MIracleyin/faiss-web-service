from clickhouse_driver import Client
from typing import List
from itertools import islice

def get_database(db_para):
    if 'host' in db_para.keys():
        return Client(**db_para)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())