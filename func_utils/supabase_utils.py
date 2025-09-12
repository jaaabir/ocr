import os
import dotenv
import numpy as np 

from PIL import Image
from io import BytesIO
from supabase import create_client, Client

def init_supabase(url: str|None = None, key: str|None = None) -> Client:
    if url is None or key is None:
        dotenv.load_dotenv(os.path.join('..', 'env'))
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_API_KEY')
    supabase: Client = create_client(url, key)
    return supabase

def retreive_data_from_table(supabase, table_name = os.environ['TABLE_NAME'], limit = -1):
    res = supabase.table(table_name).select("*")
    if limit > 0:
        res = res.limit(limit)
    return res.execute().data

retreive_data_from_bucket = lambda supabase,fname,bucket_name=os.environ['BUCKET_NAME']: supabase.storage.from_(bucket_name).download(fname)

def load_buffer_to_np(supabase, fname):
    buffer = retreive_data_from_bucket(supabase, fname)
    img = Image.open(BytesIO(buffer)).convert("RGB")
    return np.array(img, dtype="uint8")