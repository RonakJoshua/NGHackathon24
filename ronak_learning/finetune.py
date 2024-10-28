import os
import math
import utils
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from engine import train_one_epoch, evaluate
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

paths=[]
labels=[]
feature_id = []
count = 0
for dirname, _, filenames in os.walk('../imagenet/images/train'):
    for filename in filenames:
        if filename[-4:]=='JPEG':
            paths+=[(os.path.join(dirname, filename))]
            feature_id+=[(os.path.join(dirname, filename))[len("../imagenet/images/train")+1:]]
            label=dirname.split('/')[-1]
            labels+=[label]
            if os.path.isfile("../imagenet/bboxes_annotations/"+(os.path.join(dirname, filename))[len("../imagenet/images/train")+1:][:-4]+"xml") == False:
                paths.pop()
                feature_id.pop()
                labels.pop()
                count += 1
                print((os.path.join(dirname, filename))[len("../imagenet/images/train")+1:][:-4]+"xml")

# print(len(paths))
# print(count)
            
tpaths=[]
tlabels=[]
for dirname, _, filenames in os.walk('../imagenet/images/val'):
    for filename in filenames:
        if filename[-4:]=='JPEG':
            tpaths+=[(os.path.join(dirname, filename))]
            label=dirname.split('/')[-1]
            tlabels+=[label]

all_labels=os.listdir('../imagenet/images/train')

# print(len(all_labels))

from torchvision.transforms import v2 as T

# /Users/ronaksingh/Documents/Github/NGHackathon24/imagenet/bboxes_annotations/n04107743/n00141669_73.xml

def get_transform(train):
    transforms = []
    if train:
        # transforms.append(T.RandomResizedCrop(size=(224, 224), antialias=True))
        # transforms.append(T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
        # transforms.append(T.RandomPhotometricDistort(0.5))
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    # transforms.append(T.TenCrop(32)) # this is a tuple of PIL Images
    # transforms.append(T.Lambda(lambda crops: torch.stack([T.PILToTensor()(crop) for crop in crops]))),
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)




##################

def read_pascal_voc_annotation(xml_file):
    # tree = ET.parse(xml_file, parser = ET.XMLParser(encoding = 'iso-8859-5'))
    # print(xml_file)
    tree = ET.parse(xml_file)

    root = tree.getroot()

    annotations = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        label = obj.find('name').text

        annotations.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'label': label
        })

    return annotations

# print(read_pascal_voc_annotation('../imagenet/bboxes_annotations/n01440764/n01440764_1775.xml'))

# imagenet_train = ImageNet(root="../imagenet/images/train/", split='train', transform=get_transform(train=True))


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = paths
        self.annotations = feature_id
        # self.annotations = ann_paths

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(self.root, "JPGImages", self.imgs[idx])
        # annotation_path = os.path.join(self.root, "Annotations", self.annotations[idx])
        img_path = self.imgs[idx]
        annotation_path = os.path.join(self.root, "bboxes_annotations", (self.annotations[idx][:-4]+"xml"))
        img = read_image(img_path)

        # print(img_path)
        
        annotation_out = read_pascal_voc_annotation(annotation_path)

        num_objs = len(annotation_out)

        boxesList = []
        labelList = []

        for a in annotation_out:
            if a["label"] in all_labels:
                labelList += [all_labels.index(a["label"])+1]
                boxesList += [a["bbox"]]

        boxes = torch.tensor(np.array(boxesList), dtype=float)
        
        tensorLabel = torch.tensor(np.array(labelList), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = tensorLabel
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



#################


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.score_thresh = 0

    return model

model = get_model_instance_segmentation(len(all_labels)+1)

###############


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = ImageNetDataset('../imagenet', get_transform(train=True))
dataset_test = ImageNetDataset('../imagenet', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
# dataset_test = torch.utils.data.Subset(dataset_test, indices[:-50])
# dataset = torch.utils.data.Subset(dataset, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

model.to(device)
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # torch.save(model.state_dict(), f'imagenet_finetune_weights_f{epoch}.pth')
#     # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), f'imagenet_finetune_weights_{epoch}.pth')

# torch.save(model.state_dict(), 'test_finetune_weights.pth')
print("That's it!")