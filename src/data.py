from torchvision import transforms
from util.hardware import load_json

# file = load(fp='/home/msis/Work/anomalyDetector/EfficientAD/parameters.json')
# image_size = file['image_size']
# json_file = load_json(fp='/home/msis/Work/anomalyDetector/EfficientAD/parameters.json')
# image_size = json_file['image_size']

image_size = load_json(fp='/home/msis/Work/anomalyDetector/EfficientAD/parameters.json')['image_size']


basic_transform = transforms.Compose(transforms=[
                        transforms.Resize(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])