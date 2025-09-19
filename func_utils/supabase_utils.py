import os
import httpx
import dotenv
import numpy as np 

from PIL import Image
from io import BytesIO
from supabase import create_client, Client, ClientOptions

def init_supabase(url: str|None = None, key: str|None = None) -> Client:
    if url is None or key is None:
        dotenv.load_dotenv(os.path.join('..', 'env'))
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_API_KEY')
    supabase: Client = create_client(url, key, options=ClientOptions(
        httpx_client=httpx.Client(timeout=httpx.Timeout(120.0))
    ))
    return supabase

def retreive_data_from_table(supabase, table_name = None, limit = -1):
    if table_name is None:
        table_name = os.environ['TABLE_NAME']
    res = supabase.table(table_name).select("*")
    if limit > 0:
        res = res.limit(limit)
    return res.execute().data

retreive_data_from_bucket = lambda supabase,fname,bucket_name: supabase.storage.from_(bucket_name).download(fname)
list_files_in_bucket = lambda supabase, bucket_name, path, limit = 21000, get_full_path = True : [f"{path}/{i['name']}" if get_full_path else i['name'] for i in supabase.storage.from_(bucket_name).list(path, options={'limit':limit})]

def load_buffer_to_np(supabase, fname, bucket_name = None):
    if bucket_name is None:
        os.environ['BUCKET_NAME']

    try:
        buffer = retreive_data_from_bucket(supabase, fname, bucket_name)
        img = Image.open(BytesIO(buffer)).convert("RGB")
    except Exception as e:
        print(e)
        return None 
    return np.array(img, dtype="uint8")