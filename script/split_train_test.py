import os, shutil
import numpy
import scipy.io

def extract_file_and_label(annos):
    data = scipy.io.loadmat(annos)["annotations"]
    data = data.reshape(data.shape[1])
    files = []
    labels = []
    for x in data:
        labels.append(x[4])
        files.append(x[5])
    files = numpy.array(files)
    labels = numpy.array(labels)
    numpy.savez("test_annos.npz", files=files, labels=labels)

def save_images(pairs, dirname="."):
    generated_dir = dirname
    images, labels = pairs
    for i, image in enumerate(images):
        if ".jpg" in labels[i][0]: continue
        sub_dir = '{}/{:0>5}/'.format(generated_dir, str(labels[i][0]).replace('[', '').replace(']', ''))
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        img_path = str(image[0]).replace('[','').replace(']','')
        orig_path = generated_dir + "/" + img_path
        preview_path = sub_dir + img_path
        print(orig_path, preview_path)
        shutil.move(orig_path, preview_path)


train_dataset = numpy.load('train_annos.npz')
train_images = train_dataset['files']
train_labels = train_dataset['labels']
save_images((train_images, train_labels), 'train')

test_dataset = numpy.load('test_annos.npz')
test_images = test_dataset['files']
test_labels = test_dataset['labels']
save_images((test_images, test_labels), 'test')