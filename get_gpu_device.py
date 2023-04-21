import torch


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("available_gpus", available_gpus)

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)

print('current device',torch.cuda.current_device() )


for i in range(torch.cuda.device_count()):
   print("gpu prop : ", i , torch.cuda.get_device_properties(i))
   print("gpu name : ", torch.cuda.get_device_name(i))

   