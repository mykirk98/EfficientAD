from torchvision import transforms
from util.hardware import load_json

json_file = load_json(fp='/home/msis/Work/anomalyDetector/EfficientAD/parameters.json')
image_size = json_file['image_size']
channel_mean = json_file['channel_mean']
channel_std = json_file['channel_std']

basic_transform = transforms.Compose(transforms=[
                        transforms.Resize(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=channel_mean, std=channel_std)
                    ])

penalty_transform = transforms.Compose(transforms=[
                        transforms.Resize(size=(2*image_size, 2*image_size)),
                        transforms.RandomGrayscale(p=0.3),
                        transforms.CenterCrop(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=channel_mean, std=channel_std)
])