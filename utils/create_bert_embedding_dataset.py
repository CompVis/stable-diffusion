import webdataset as wds
from tqdm import tqdm
from utils.sequence_embedding import encode_sequence_bert, BERT_FEATURE_LENGTH
import numpy as np

url = "/data/wei/hpa-webdataset-all-composite/webdataset_info.tar"
dataset = wds.WebDataset(url).decode().to_tuple("__key__", "info.json")
with open("error-log-bert.txt", "w") as log:
    with wds.TarWriter('/data/wei/hpa-webdataset-all-composite/webdataset_bert.tar') as sink:
        for idx, data in tqdm(enumerate(dataset)):
            info = data[1]
            if info["sequences"]:
                try:
                    bert_embedding = encode_sequence_bert(info["sequences"][0], device="cuda", max_length=8192)[1].cpu().numpy()
                except Exception as e:
                    log.write(f"Failed to run bert for {info}\n")
                    print(e, info)
            else:
                bert_embedding = np.zeros([BERT_FEATURE_LENGTH], dtype='float32')
            sink.write({
                "__key__": data[0],
                "bert.pyd": bert_embedding
            })


"""
error:

10183it [02:27, 69.20it/s]
Traceback (most recent call last):
  File "create_bert_embedding_dataset.py", line 14, in <module>
    bert_embedding = encode_sequence_bert(info["sequences"][0], device="cuda")[1].cpu().numpy()
  File "/home/wei.ouyang/workspace/stable-diffusion/sequence_embedding.py", line 62, in encode_sequence_bert
    protein_vector = torch.tensor(vec[None, :]).to(device)
RuntimeError: CUDA error: device-side assert triggered

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "create_bert_embedding_dataset.py", line 14, in <module>
    bert_embedding = encode_sequence_bert(info["sequences"][0], device="cuda")[1].cpu().numpy()
KeyboardInterrupt
"""