import resnet as r
from time import perf_counter

start_time = perf_counter()
log = open('/home/ubuntu/pytorch-alloc-hookup/hookup_scripts/models/pytorchLog', 'a')
log.write('==============torch imported============\n')
log.flush()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


log.write('==============model load begin============\n')
log.flush()
model = r.resnet18()
log.write('==============model load end============\n')
log.flush()
model.eval()
log.write('==============model eval end============\n')
log.flush()
model(input_batch) 

log.close()

end_time = perf_counter()
print(end_time - start_time)
