# -*- coding: utf-8 -*-
"""Interacting with CLIP.ipynb의 사본

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bnUa-kmNg7B6LJ9QsyhNDJNcZV3MuYZ1

# Interacting with CLIP

This is a self-contained notebook that shows how to download and run CLIP models, calculate the similarity between arbitrary image and text inputs, and perform zero-shot image classifications.

# Preparation for Colab

Make sure you're running a GPU runtime; if not, select "GPU" as the hardware accelerator in Runtime > Change Runtime Type in the menu. The next cells will install the `clip` package and its dependencies, and check if PyTorch 1.7.1 or later is installed.
"""

# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git

import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""# Loading the model

`clip.available_models()` will list the names of available CLIP models.
"""

import clip

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

"""# Image Preprocessing

We resize the input images and center-crop them to conform with the image resolution that the model expects. Before doing so, we will normalize the pixel intensity using the dataset mean and standard deviation.

The second return value from `clip.load()` contains a torchvision `Transform` that performs this preprocessing.


"""

preprocess

"""# Text Preprocessing

We use a case-insensitive tokenizer, which can be invoked using `clip.tokenize()`. By default, the outputs are padded to become 77 tokens long, which is what the CLIP models expects.
"""

clip.tokenize("Hello World!")

clip.tokenize("world Hello !")

"""# Setting up input images and texts

We are going to feed 8 example images and their textual descriptions to the model, and compare the similarity between the corresponding features.

The tokenizer is case-insensitive, and we can freely give any suitable textual descriptions.
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# images in skimage to use and their textual descriptions
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}

os.listdir(skimage.data_dir)

original_images = []
images = []
texts = []
plt.figure(figsize=(16, 5))

for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image)) # images : list, [[3, 224, 224], [3, 224, 224], ..., [3, 224, 224]]
    texts.append(descriptions[name])

# print(original_images) # original image (RGB)
# print(images) # preprocessing 을 거친 image
# print(texts) # description of texts
plt.tight_layout()

"""## Building features

We normalize the images, tokenize each text input, and run the forward pass of the model to get the image and text features.
"""

image_input = torch.tensor(np.stack(images)).cuda() # shape: [8, 3, 224, 224], type: torch.tensor
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda() # shape : [8, 77]

print(image_input.shape)
print(text_tokens.shape)

with torch.no_grad():
    image_features = model.encode_image(image_input).float() # shape : [8, 512]
    text_features = model.encode_text(text_tokens).float() # shape : [8, 512]

"""## Calculating cosine similarity

We normalize the features and calculate the dot product of each pair.
"""

image_features /= image_features.norm(dim=-1, keepdim=True) # [8, 512]
text_features /= text_features.norm(dim=-1, keepdim=True) # [8, 512]
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T  # shape : [8, 8]

print(similarity)
print(similarity.shape)

count = len(descriptions) # count = 8, discriptions are below;
                          # {'page': 'a page of text about segmentation', 
                          # 'chelsea': 'a facial photo of a tabby cat', 
                          # 'astronaut': 'a portrait of an astronaut with the American flag', 
                          # 'rocket': 'a rocket standing on a launchpad', 
                          # 'motorcycle_right': 'a red motorcycle standing in a garage', 
                          # 'camera': 'a person looking at a camera on a tripod', 
                          # 'horse': 'a black-and-white silhouette of a horse', 
                          # 'coffee': 'a cup of coffee on a saucer'}

plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)  # vmin, vmax 는 colorbar 의 범위를 조정함. 이는 그냥 시각화의 도구일뿐임. 아래 출력에서는 숫자들의 바탕색에 해당
# plt.colorbar()
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower") # 상단 이미지 부분에 해당
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
        pass

for side in ["left", "top", "right", "bottom"]:
  plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=20)

"""# Zero-Shot Image Classification

You can classify images using the cosine similarity (times 100) as the logits to the softmax operation.
"""

from torchvision.datasets import CIFAR100

cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)

print(cifar100)

text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
print(text_descriptions) # 100개의 라벨
text_tokens = clip.tokenize(text_descriptions).cuda()  # shape : [100, 77] (100개의 descriptions 에 대해서 max length 77로 tokenize)
print(text_tokens)
print(text_tokens.shape)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float() # shape : [100, 512] 512 features 를 가진 100개의 문장
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(text_features.shape) # shape of text_features = [100, 512]
print(100.0 * image_features)  # shape of image_features = [8, 512]
text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(text_probs.shape)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1) # topk : tensor 를 input 으로 넣으면 그 중 가장 큰 값 n 개를 반환

print(top_probs)
print(top_probs.shape)
print(top_labels)
print(top_labels.shape)

plt.figure(figsize=(16, 16))

for i, image in enumerate(original_images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace=0.5)
plt.show()