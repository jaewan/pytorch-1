import torch

log = open("/home/ubuntu/pytorchLog", "a")
log.write("=============import done======================\n")
log.flush()

torch.empty((2,3), True)
