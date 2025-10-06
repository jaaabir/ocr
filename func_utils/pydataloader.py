import os
import cv2
import json
import torch 
import dotenv
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from .supabase_utils import init_supabase, load_buffer_to_np, retreive_data_from_table, list_files_in_bucket
from pathlib import Path


get_basename = lambda f: os.path.basename(f)
remove_ext = lambda f: os.path.splitext(get_basename(f))[0] if isinstance(f, str) else f
str_to_json = lambda f: json.loads(f) if isinstance(f, str) else f

def get_val_from_json(data, val_name, root_key_name = None):
    val = [c.get(val_name, []) for c in (data[root_key_name] if root_key_name else data)]
    return val

class TableDataset(Dataset):
    def __init__(self, metadata, transform=False, img_size=224, n_channels=2, channel_first=True, to_torch=True, env_path = '.'):
        """
        Args:
            metadata (list[dict]): a single row contains [id, filename, document structure metadata, text metadata]
            transform (callable, optional): Transformations for image preprocessing
            img_size (int): resize target for images if no transform given
        """
        dotenv.load_dotenv(env_path)
        self.SUPABASE = init_supabase()
        self.data = metadata
        self.transform = transform
        self.img_size = img_size
        self.n_channels = n_channels
        self.channel_first = channel_first
        self.to_torch = to_torch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # --- read image ---
        img_path = self.data[idx]['name'] + '.jpg'
        struct = self.data[idx]['struct']
        words = self.data[idx]['words']
        img = load_buffer_to_np(self.SUPABASE, img_path)

        if self.n_channels == 2 and len(img.shape) == 3:
            # Convert RGB to grayscale if image has 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_h, img_w = img.shape[:2]


        # resize if no external transform is supplied
        if self.transform:
            img, bboxes = self.transform(img, struct, words)
        else:
            sbboxes = get_val_from_json(struct, "bbox", 'objects')
            wbboxes = get_val_from_json(words, "bbox")
            bboxes = sbboxes + wbboxes
            img = img.astype(np.float32) / 255.0
        
        if self.img_size:
            img = cv2.resize(img, self.img_size)

        if self.to_torch:
            img = torch.from_numpy(img)

        if self.n_channels == 3 and self.channel_first:
            img = img.permute(2, 0, 1)  # HWC -> CHW

        sample = {
            "image": img,
            "struct": struct,
            "words": words,
            'texts' : get_val_from_json(words, 'text'),
            'bboxes' : bboxes
        }

        return sample
    

class ImageDataset(Dataset):
    def __init__(self, image_path, transform=False, n_channels=2, channel_first=True, to_torch=True, img_size = (864, 864)):
        self.data = image_path
        self.transform = transform
        self.n_channels = n_channels
        self.channel_first = channel_first
        self.to_torch = to_torch
        self.H, self.W = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = cv2.imread(image_path)

        if self.n_channels == 2 and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image, None, None, self.H, self.W)

        if self.to_torch:
            image = torch.from_numpy(image)
        
            if self.channel_first:
                image = image.permute(-1, 0, 1)

        return image
    
def read_json(fname, encoding='utf-8'):
    data = []

    # Open the JSONL file and read each line
    with open(fname, 'r', encoding=encoding) as f:
        for line in f:
            try:
                # Parse each line as a JSON object
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error: {e}")
    return data


import cv2
import numpy as np

def render_text_image(h: int, w: int, text: str, 
                      font_scale: float = 1.0, thickness: int = 2,
                      bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    # Create a blank image with given background color
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get boundary of text
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Center the text
    x = (w - text_w) // 2
    y = (h + text_h) // 2

    # Render the text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img


def get_random_word(lang):
    """
    Selects a random word from a corpus.

    Args:
        corpus (list[str] or str): If list, picks directly from it. 
                                   If str, treats it as a path to a text file (one word per line or space-separated).

    Returns:
        str: A randomly selected word.
    """
    words = []

    corpus = 'eng_sample_corpus.txt' if lang == 'en' else 'pt_sample_corpus.txt'

    # If corpus is a filepath (string), read words
    if isinstance(corpus, str):
        with open(corpus, "r", encoding="utf-8") as f:
            text = f.read()
            # Split by whitespace
            words = text.split()
    elif isinstance(corpus, list):
        words = corpus
    else:
        raise TypeError("Corpus must be a list of words or a filepath (str).")

    if not words:
        raise ValueError("Corpus is empty!")

    return ' '.join(np.random.choice(words, 2))

# Example usage
if __name__ == "__main__":
    img = render_text_image(300, 600, "Hello OCR!", font_scale=2.0, thickness=3,
                            bg_color=(255, 255, 255), text_color=(0, 0, 0))
    cv2.imwrite("output_text.png", img)



get_language = lambda x, from_supabase = False : Path(x).parts[0 if from_supabase else 2].split('_')[-1]
get_split = lambda x : x.split('\\')[3]
local_fpath_to_supbase_fpath = lambda path : f"SynthDoG_{get_language(path)}/{get_split(path)}/{get_basename(path)}" 

class SynthDogDataset(Dataset):
    def __init__(self, image_path = None, output_jsons_path = None, image_feature_extractor = None, text_tokenizer = None, max_token_size = 512, n_channels = 3, 
                 return_processed_outputs = True, required_input_ids=False, sample_size = -1, read_images_from_supabase = False, split='train', 
                 env_path=os.path.join(os.getcwd(), '.env'), img_size = (224,224), seed = -1):
        # self.data = image_path if not read_images_from_supabase else list_files_in_bucket(self.supabase, self.bucket_name, path)
        if read_images_from_supabase:
            dotenv.load_dotenv(env_path)
            self.supabase = init_supabase(os.environ['COMU_SUPABASE_URL'], os.environ['COMU_SUPABASE_API_KEY'])
            self.bucket_name = os.environ['COMU_BUCKET_NAME']
            en = list_files_in_bucket(self.supabase, self.bucket_name, f'SynthDog_en/{split}')
            pt = list_files_in_bucket(self.supabase, self.bucket_name, f'SynthDog_pt/{split}')
            self.data = en + pt
        else:
            self.data = image_path

        self.metadata = [read_json(output_json_path) for output_json_path in output_jsons_path]
        self.image_feature_extractor = image_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.max_token_size = max_token_size
        self.n_channels = n_channels
        self.return_processed_outputs = return_processed_outputs
        self.required_input_ids = required_input_ids
        self.total_languages = len(output_jsons_path)
        self.read_images_from_supabase = read_images_from_supabase
        self.H, self.W = img_size
        self.seed = seed

        if sample_size > 0:
            print(self.data[:2])
            self.data = self.sample_data_equally(sample_size, self.data)
        
        self.json_metadata = {}

        for i in range(len(self.metadata)): 
            language = get_language(output_jsons_path[i])
            for jdata in self.metadata[i]:
                fname = jdata['file_name'] + '_' + language
                gt = jdata['ground_truth']
                self.json_metadata[fname] = gt

        print(f"Length of _.images: {len(self.data)} | Length of _.json_metadata: {len(self.json_metadata)}")

    def __len__(self):
        return len(self.data)
    
    def sample_data_equally(self, total_n_samples, paths):
        if self.seed > -1:
            np.random.seed(self.seed)
        image_paths = np.random.shuffle(paths)
        n_sample_per_lang = total_n_samples // self.total_languages
        sampled_paths = []
        sampled_lang_counter = {}
        i = 0

        while len(sampled_paths) < total_n_samples:
            if i >= len(paths):
                break
            curr_lang = get_language(paths[i], self.read_images_from_supabase)
            if curr_lang not in sampled_lang_counter:
                sampled_lang_counter[curr_lang] = 0
            
            if sampled_lang_counter[curr_lang] < n_sample_per_lang:
                sampled_paths.append(paths[i])
                sampled_lang_counter[curr_lang] += 1
            i += 1
        print(f"Sampled lang counter: {sampled_lang_counter}")
        return sampled_paths


    def __getitem__(self, idx):
        image_path = self.data[idx] 
        language = get_language(image_path, self.read_images_from_supabase)
        fname = get_basename(image_path)
        image = cv2.imread(image_path) if not self.read_images_from_supabase else load_buffer_to_np(self.supabase, image_path, self.bucket_name)
        kname = fname + '_' + language
        metadata = self.json_metadata.get(kname)
        target_text = json.loads(metadata).get('gt_parse').get('text_sequence')

        if image is None:
            print("Image is none, generating a sample image and text...")
            print(f"Image path: {image_path}")
            target_text = get_random_word(language)
            image = render_text_image(self.H, self.W, target_text)

        if self.n_channels == 2 and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.text_tokenizer:
            image_features = self.image_feature_extractor(image, return_tensors="pt").pixel_values.squeeze(0)
            tokenizer_op = self.text_tokenizer(target_text)
            attention_mask = torch.tensor(tokenizer_op['attention_mask'])
            output_tokens = tokenizer_op['input_ids']
            if output_tokens[0] != self.text_tokenizer.bos_token_id:
                output_tokens[0] = self.text_tokenizer.bos_token_id
            output_tokens = torch.tensor(output_tokens)
        else:
            proc_outputs = self.image_feature_extractor(
                images=image,
                text = target_text,
                truncation=False,
                # padding="max_length",
                # max_length=self.max_token_size,
                return_tensors="pt",
            )
            image_features = proc_outputs["pixel_values"].squeeze(0)
            output_tokens = proc_outputs["input_ids"].squeeze(0)
            attention_mask = proc_outputs["attention_mask"].squeeze(0)

        data =  {
        'pixel_values'   : image_features if self.return_processed_outputs else image,
        'labels'         : output_tokens if self.return_processed_outputs else target_text,
        'image'          : image,
        'text'           : target_text,
        'image_path'     : image_path,
        'metadata_fname' : fname
    }
        if self.return_processed_outputs:
            data['attention_mask'] = attention_mask
        return data
        
