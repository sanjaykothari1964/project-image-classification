# project-image-classification
Image Classification
In this project, you'll classify images from the CIFAR-10 dataset
(https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of airplanes, dogs, cats, and other
objects. You'll preprocess the images, then train a convolutional neural network on all the samples.
The images need to be normalized and the labels need to be one-hot encoded. You'll get to apply
what you learned and build a convolutional, max pooling, dropout, and fully connected layers. At the
end, you'll get to see your neural network's predictions on the sample images.
Get the Data
Run the following cell to download the CIFAR-10 dataset for python
(https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
Data
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the
80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10
object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and
Geoffrey Hinton.
Let's get the data by running the following function
Image_classification

3/1/2018

http://localhost:8888/notebooks/3rd%20and%204th%20March%20classes/Project_2/Project%202%20image_classification.ipynb 2/7
In [2]:

Explore the Data
The dataset is broken into batches to prevent your machine from running out of memory. The
CIFAR-10 dataset consists of 5 batches, named data_batch_1, data_batch_2, etc.. Each batch
contains the labels and images that are one of the following:
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
Understanding a dataset is part of making predictions on the data. Play around with the code cell
below by changing the batch_id and sample_id. The batch_id is the id for a batch (1-5). The
sample_id is the id for a image and label pair in the batch.
Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are
the labels in order or random?". Answers to questions like these will help you preprocess the data
and end up with better predictions.
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
cifar10_dataset_folder_path = 'cifar-10-batches-py'
class DLProgress(tqdm):
last_block = 0
def hook(self, block_num=1, block_size=1, total_size=None):
self.total = total_size
self.update((block_num - self.last_block) * block_size)
self.last_block = block_num
if not isfile('cifar-10-python.tar.gz'):
with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset
urlretrieve(
'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
'cifar-10-python.tar.gz',
pbar.hook)
if not isdir(cifar10_dataset_folder_path):
with tarfile.open('cifar-10-python.tar.gz') as tar:
tar.extractall()
tar.close()

Image_classification

3/1/2018

http://localhost:8888/notebooks/3rd%20and%204th%20March%20classes/Project_2/Project%202%20image_classification.ipynb 3/7
The following are some helper functions students can use
in their code

Image_classification

3/1/2018

http://localhost:8888/notebooks/3rd%20and%204th%20March%20classes/Project_2/Project%202%20image_classification.ipynb 4/7
In [3]: import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

def _load_label_names():
"""
Load the label names from file
"""
return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
"""
Load a batch of the dataset
"""
with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode=
batch = pickle.load(file, encoding='latin1')
features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0
labels = batch['labels']
return features, labels
def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
"""
Display Stats of the the dataset
"""
batch_ids = list(range(1, 6))
if batch_id not in batch_ids:
print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
return None
features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
if not (0 <= sample_id < len(features)):
print('{} samples in batch {}. {} is out of range.'.format(len(features)
return None
print('\nStats of batch {}:'.format(batch_id))
print('Samples: {}'.format(len(features)))
print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True
print('First 20 Labels: {}'.format(labels[:20]))
sample_image = features[sample_id]
sample_label = labels[sample_id]
label_names = _load_label_names()
print('\nExample of Image {}:'.format(sample_id))
print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_
print('Image - Shape: {}'.format(sample_image.shape))
print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_
plt.axis('off')

Image_classification

3/1/2018

http://localhost:8888/notebooks/3rd%20and%204th%20March%20classes/Project_2/Project%202%20image_classification.ipynb 5/7

plt.imshow(sample_image)

def
"""
Preprocess data and save it to file
"""
features = normalize(features)
labels = one_hot_encode(labels)
pickle.dump((features, labels), open(filename, 'wb'))
def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encod
"""
Preprocess Training and Validation Data
"""
n_batches = 5
valid_features = []
valid_labels = []
for batch_i in range(1, n_batches + 1):
features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i
validation_count = int(len(features) * 0.1)
# Prprocess and save a batch of training data
_preprocess_and_save(
normalize,
one_hot_encode,
features[:-validation_count],
labels[:-validation_count],
'preprocess_batch_' + str(batch_i) + '.p')
# Use a portion of training batch for validation
valid_features.extend(features[-validation_count:])
valid_labels.extend(labels[-validation_count:])
# Preprocess and Save all validation data
_preprocess_and_save(
normalize,
one_hot_encode,
np.array(valid_features),
np.array(valid_labels),
'preprocess_validation.p')
with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
batch = pickle.load(file, encoding='latin1')
# load the training data
test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpo
test_labels = batch['labels']
# Preprocess and Save all training data
_preprocess_and_save(
normalize,
one_hot_encode,
np.array(test_features),

Image_classification

_preprocess_and_save(normalize, one_hot_encode, features, labels, filename):

3/1/2018

http://localhost:8888/notebooks/3rd%20and%204th%20March%20classes/Project_2/Project%202%20image_classification.ipynb 6/7

np.array(test_labels),
'preprocess_training.p')
def batch_features_labels(features, labels, batch_size):
"""
Split features and labels into batches
"""
for start in range(0, len(features), batch_size):
end = min(start + batch_size, len(features))
yield features[start:end], labels[start:end]
def load_preprocess_training_batch(batch_id, batch_size):
"""
Load the Preprocessed Training data and return them in batches of <batch_size
"""
filename = 'preprocess_batch_' + str(batch_id) + '.p'
features, labels = pickle.load(open(filename, mode='rb'))
# Return the training data in batches of size <batch_size> or less
return batch_features_labels(features, labels, batch_size)
def display_image_predictions(features, labels, predictions):
n_classes = 10
label_names = _load_label_names()
label_binarizer = LabelBinarizer()
label_binarizer.fit(range(n_classes))
label_ids = label_binarizer.inverse_transform(np.array(labels))
fig, axies = plt.subplots(nrows=4, ncols=2)
fig.tight_layout()
fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)
n_predictions = 3
margin = 0.05
ind = np.arange(n_predictions)
width = (1. - 2. * margin) / n_predictions
for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip
pred_names = [label_names[pred_i] for pred_i in pred_indicies]
correct_name = label_names[label_id]
axies[image_i][0].imshow(feature*255)
axies[image_i][0].set_title(correct_name)
axies[image_i][0].set_axis_off()
axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
axies[image_i][1].set_yticks(ind + margin)
axies[image_i][1].set_yticklabels(pred_names[::-1])
axies[image_i][1].set_xticks([0, 0.5, 1.0])

Image_classification

3/1/2018

http://localhost:8888/notebooks/3rd%20and%204th%20March%20classes/Project_2/Project%202%20image_classification.ipynb 7/7
In [21]:

Stats of batch 3:
Samples: 10000
Label Counts: {0: 994, 1: 1042, 2: 965, 3: 997, 4: 990, 5: 1029, 6: 978, 7: 101
5, 8: 961, 9: 1029}
First 20 Labels: [8, 5, 0, 6, 9, 2, 8, 3, 6, 2, 7, 4, 6, 9, 0, 0, 7, 3, 7, 2]
Example of Image 5:
Image - Min Value: 9 Max Value: 255
Image - Shape: (32, 32, 3)
Label - Label Id: 2 Name: bird
