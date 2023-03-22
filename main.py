import torch
import torchvision
import torchvision.models.detection
from torchvision import transforms as T

from PIL import Image
import cv2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()