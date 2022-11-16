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
      target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
      params:
        seed: 123
        train_split: 0.95
        group: 'train'
        cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-1000.pickle
        channels: [1, 1, 1]
        return_info: true
        filter_func: has_location
    validation:
      target: ldm.data.hpa.HPACombineDatasetMetadataInMemory
      params:
        seed: 123
        train_split: 0.95
        group: 'validation'
        cache_file: /data/wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256-1000.pickle
        channels: [1, 1, 1]
        return_info: true
        filter_func: has_location
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
for d in data.datasets['validation']:
  print(d['info']['Ab state'], d['info']['locations'])
  # if d['location_caption'] == 'nan':
  #   print('.')
# d = data.datasets['validation'][0]
# print(d)
