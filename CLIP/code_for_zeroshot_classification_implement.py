import os
import torch
import clip
import numpy as np

from torchvision.datasets import CIFAR100, CIFAR10

print(clip.available_models())
model, preprocess = clip.load("ViT-B/16")
# print(preprocess)

dir_path = os.path.dirname(__file__)
# cifar100 = CIFAR100(f"{dir_path}/CIFAR/test/", transform=preprocess, download=True, train=False)
cifar100 = CIFAR100(f"{dir_path}/CIFAR/test/", download=False, train=False)
images = []
ground_truth_labels = []
classes = [label for label in cifar100.classes]
labels = clip.tokenize(["The photo of "+label for label in cifar100.classes])
print(labels.shape)
for idx, (data, label) in enumerate(cifar100):
    if 9000 <= idx <= 9999:
        pass
    else:
        continue
    images.append(preprocess(data))
    ground_truth_labels.append([label, classes[label]])
image_input = torch.tensor(np.stack(images))
print(image_input.shape)

with torch.no_grad():
    image_features = model.encode_image(image_input).float() # shape : [1000, 512]
    text_features = model.encode_text(labels).float() # shape : [100, 512]
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
print(image_features.shape) 
print(text_features.shape)

'''
# zero shot classification
'''

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # [10000, 100]
print(text_probs.shape)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1) # topk : tensor 를 input 으로 넣으면 그 중 가장 큰 값 n 개를 반환

print(top_probs)
print(top_probs.shape)
correct = 0
for idx, x in enumerate(top_labels):
    index = x.item()
    print("ground_truth: ",ground_truth_labels[idx])
    print("prediction: ",index, classes[index])
    print()
    if ground_truth_labels[idx][0] == index:
        correct += 1
accuracy = correct * 100 / len(ground_truth_labels)
print(str(accuracy)+"%")

with open(f"{dir_path}/test_results.txt", encoding="utf-8", mode='a') as f:
    f.write("\n")
    f.writelines(str(correct))
    

