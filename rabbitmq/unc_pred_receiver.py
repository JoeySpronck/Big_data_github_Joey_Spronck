### LIBRARIES
import torch, torchvision
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pika

### IMPORT PREDTICITON MODEL
import models

model = models.CNN()
model.load_state_dict(torch.load("../model_folder/model_2_best_1.pt"))

### IMPORT UNCERTAINTY MODEL
u_model = pickle.load(open("../model_folder/rf_uncertainty_model.sav", "rb"))

### IMPORT DATA
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# mnist_data_train = torchvision.datasets.MNIST('mnist_data',
#                                               transform=T,
#                                               download=True,
#                                               train=True)
mnist_data_valid = torchvision.datasets.MNIST('mnist_data',
                                              transform=T,
                                              download=True,
                                              train=False)
# emnist_data_train = torchvision.datasets.EMNIST('emnist_data',
#                                                split = "balanced",
#                                                transform=T,
#                                                download=True,
#                                                train=True)
emnist_data_valid = torchvision.datasets.EMNIST('emnist_data',
                                                split="balanced",
                                                transform=T,
                                                download=True,
                                                train=False)



connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='map')


def callback(ch, method, properties, body):
    ### TO BE INPUT FROM SENDER
    I = int(body)
    sample = mnist_data_valid[I]
    image, label = sample

    ### NORMAL PREDICTION
    model.eval()
    with torch.no_grad():
        pred_eval = model(image.unsqueeze(dim=0))
    model.train()
    pred_eval_s = F.softmax(pred_eval, dim=1)
    prediction = pred_eval_s.argmax(dim=1).item()

    ### MC PREDICTION
    n = 100
    mc_list = []
    with torch.no_grad():
        for i in range(n):
            preds = model(image.unsqueeze(dim=0))
            mc_list.append(preds.unsqueeze(dim=2))

    mc = torch.cat((mc_list), dim=2)
    mc_s = F.softmax(mc, dim=1)

    ### MC VALUES
    image_values = []
    for _image in mc_s:
        target_values = []
        for _target in _image:
            target_values.append(_target.mean())
            target_values.append(_target.std())
        image_values.append(target_values)

    mc_tensor = torch.tensor(image_values)
    mc_features = torch.cat((mc_tensor, pred_eval_s), dim=1)
    X = mc_features

    new_X_2_list = []
    for i in range(len(X)):
        index_2 = reversed(X[i][20:30].argsort())[:5]
        keep_values_2 = torch.cat((X[i][20:30][index_2], X[i][index_2 * 2], X[i][index_2 * 2 + 1]))
        new_X_2_list.append(keep_values_2.unsqueeze(dim=0))
    new_X_2 = torch.cat(new_X_2_list, dim=0)

    ### PREDICT UNCERTAINTY
    uncertainty = int(u_model.predict(new_X_2))

    ### PRINT OUTCOME
    print(f"INDEX: {int(body)}\n    Prediction label:  {prediction}\n    Uncertainty label: {uncertainty}")
    # print(" [x] Received %r" % body)
    # return body

channel.basic_consume(
    queue='map', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()



