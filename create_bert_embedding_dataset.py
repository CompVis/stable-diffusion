import webdataset as wds
from tqdm import tqdm
from sequence_embedding import encode_sequence_bert, BERT_FEATURE_LENGTH
import numpy as np

url = "/data/wei/hpa-webdataset-all-composite/webdataset_info.tar"
dataset = wds.WebDataset(url).decode().to_tuple("__key__", "info.json")
with wds.TarWriter('/data/wei/hpa-webdataset-all-composite/webdataset_bert.tar') as sink:
    for idx, data in tqdm(enumerate(dataset)):
        info = data[1]
        if info["sequences"]:
            bert_embedding = encode_sequence_bert(info["sequences"][0], device="cuda")[1].cpu().numpy()
        else:
            bert_embedding = np.zeros([BERT_FEATURE_LENGTH], dtype='float32')
        sink.write({
            "__key__": data[0],
            "bert.pyd": bert_embedding
        })