import torch
import ray
import argparse
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--NUM_TRIAL', '-tr', type=int, default=1)
parser.add_argument('--NUM_TASKS', '-t', type=int, default=1)
args = parser.parse_args()
params = vars(args)

NUM_TRIAL = params['NUM_TRIAL']
NUM_TASKS = params['NUM_TASKS']

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

def warmup():
    @ray.remote
    def warm():
        a = 1
        for i in range(1000000):
            a = a+1
        return True
    refs = [warm.remote() for _ in range(16)]
    ray.get(refs)
    return

def test_resnet(filename, RESNET_VER):
    @ray.remote(num_cpus=1)
    def resnet(filename, RESNET_VER):
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

        model = torch.hub.load('pytorch/vision:v0.10.0', RESNET_VER, pretrained=True)
        model.eval()
        start_time = perf_counter()
        with torch.no_grad():
            output = model(input_batch)
        end_time = perf_counter()
        return end_time - start_time
    refs = [resnet.remote(filename, RESNET_VER) for _ in range(NUM_TASKS)]
    results = ray.get(refs)
    return sum(results)/NUM_TASKS
    

times = []
RESNET_VER = 'resnet152'
ray.init()
warmup()

for i in range(NUM_TRIAL):
    times.append(test_resnet(filename, RESNET_VER))
print(sum(times)/NUM_TRIAL)

'''
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
'''
