# AltCLIP for Huggingface

我们已经上传了模型权重到 `transformers` ，只需要几行代码就能快速使用我们的模型！ [Huggingface Model Card](https://huggingface.co/BAAI/AltCLIP)

we have uploaded our model to `transformers`. you can use our model by a few lines of code. If you find it useful, feel free to star🌟!


# requirements

我们在以下环境进行了测试，请尽量保证包版本符合要求。

```
transformeres >= 4.21.0
```
# Inference Code

```python

from PIL import Image
import requests

# transformers version >= 4.21.0
from modeling_altclip import AltCLIP
from processing_altclip import AltCLIPProcessor

# now our repo's in private, so we need `use_auth_token=True`
model = AltCLIP.from_pretrained("BAAI/AltCLIP")
processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

```