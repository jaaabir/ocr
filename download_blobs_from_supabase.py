import os
import dotenv
from tqdm import tqdm
from func_utils.supabase_utils import list_files_in_bucket, retreive_data_from_bucket, init_supabase

def download_files_from_bucket(supabase, bucket_name: str, path: str, local_base_path: str):
    """
    Download all files from a specified bucket path and store them locally
    under the same folder structure.

    Args:
        supabase: Supabase client instance.
        bucket_name (str): Name of the Supabase storage bucket.
        path (str): Path inside the bucket to fetch files from.
        local_base_path (str): Local base folder to save files.
    """
    files = list_files_in_bucket(supabase, bucket_name, path)
    print(f"Found {len(files)} files in bucket '{bucket_name}' under '{path}'")
    for file_path in tqdm(files, desc='Downloading files: '):
        try:
            # Download file from Supabase
            data = retreive_data_from_bucket(supabase, file_path, bucket_name)

            # Create local path
            local_path = os.path.join(local_base_path, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Write file
            with open(local_path, "wb") as f:
                f.write(data)

            print(f"Downloaded: {file_path} -> {local_path}")
            break

        except Exception as e:
            print(f"Failed to download {file_path}: {e}")


def main(api, url, bucket_name):
    SUPABASE = init_supabase(url, api_key)
    Langs = ['SynthDog_en', 'SynthDog_pt']
    Splits = ['train', 'validation', 'test']
    local_root = os.path.join('synthdog', 'outputs_ol')

    for lang in Langs:
        for split in Splits:
            supabase_path = f"{lang}/{split}"
            download_files_from_bucket(SUPABASE, bucket_name, supabase_path, local_root)

if __name__ == "__main__":
    dotenv.load_dotenv()
    api_key = os.environ['COMU_SUPABASE_API_KEY']
    url = os.environ['COMU_SUPABASE_URL']
    BUCKET_NAME = os.environ['COMU_BUCKET_NAME']
    if api_key and url and BUCKET_NAME:
        main(api_key, url, BUCKET_NAME)
    else:
        print(api_key)
        print(url)
        print(BUCKET_NAME)
