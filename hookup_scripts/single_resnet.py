import torch
import argparse
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--NUM_TRIAL', '-t', type=int, default=50)
args = parser.parse_args()
params = vars(args)

NUM_TRIAL = params['NUM_TRIAL']

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

def test_resnet(model):
    model.eval()
    start_time = perf_counter()
    with torch.no_grad():
        output = model(input_batch)
    end_time = perf_counter()
    return end_time - start_time

times = []
RESNET_VER = 'resnet152'
model = torch.hub.load('pytorch/vision:v0.10.0', RESNET_VER, pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

for i in range(NUM_TRIAL):
    times.append(test_resnet(model))
print(sum(times)/NUM_TRIAL)

'''
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
'''
