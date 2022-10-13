from ldm.util import instantiate_from_config
import yaml

data_config_yaml = """
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 48
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.hpa.HPACombineDatasetMetadata
      params:
        # filename: shard-{000000..000244}
        # size: 256
        include_metadata: true
        # degradation: pil_nearest
    validation:
      target: ldm.data.hpa.HPACombineDatasetMetadata
      params:
        # filename: shard-{000245..000345}
        # size: 256
        include_metadata: true
        # degradation: pil_nearest
"""

config = yaml.safe_load(data_config_yaml)
data_config = config['data']

# data
data = instantiate_from_config(data_config)
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data.prepare_data()
data.setup()
print("#### Data #####")
for k in data.datasets:
    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

# each image is:
# 'image': array(...)
# 'file_path_': 'data/celeba/data256x256/21508.jpg'
d = data.datasets['validation'][0]
print(d['image'].shape, d['image'].max(), d['image'].min(), d["class_label"])
