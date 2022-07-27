import json
import logging
import sys
import os
import base64
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,models
from io import BytesIO


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
def Net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False #Freezing internal layers of the pre-trainned model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,10) #Setting the first fully connected layer that will be trainned
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining model and loading weights to it.
def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    #print(request_body)
    #Trasnform idiom
    transform = transforms.Compose([transforms.Resize((400, 400)),
                                     #transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
    
    encoded_im = json.loads(request_body)["inputs"]
    
    #decode 
    im = PIL.Image.open(BytesIO(base64.b64decode(encoded_im)))
    
    #Process data
    data = transform(im).unsqueeze(0)
    return data


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    class_dict = {0 : "safe driving",
              1 : "texting - right",
              2 : "talking on the phone - right",
              3 : "texting - left",
              4 : "talking on the phone - left",
              5 : "operating the radio",
              6 : "drinking",
              7 : "reaching behind",
              8 : "hair and makeup",
              9 : "talking to passenger"}
    proba = nn.Softmax(dim=1)(predictions)
    proba = [round(float(elem),4) for elem in proba[0]]
    res = {'pred_list':proba,#.cpu().numpy().tolist(),
           'pred_class': class_dict[proba.index(max(proba))],
           'confidence': max(proba)}
    return json.dumps(res)