from __future__ import division
from __future__ import print_function

import json
import panoramasdk
import cv2
import numpy as np
import boto3
from imagenet_classes import get_classes

def preprocess(img, resize_short=256, crop_size=224,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    """
    # resize to resize short
    # find the short size, 
    width = img.shape[0]
    height = img.shape[1]

    height_is_short = int(width > height) #

    if height_is_short:
        width = int(width * (resize_short/height))
        height = resize_short
    else:
        height = int(height * (resize_short/width))
        width = resize_short
    
    img = cv2.resize(img, (height, width))
    
    # center crop
    xmin = int(width/2 - crop_size/2)
    xmax = int(width/2 + crop_size/2)
    ymin = int(height/2 - crop_size/2)
    ymax = int(height/2 + crop_size/2)
    
    img = img[xmin:xmax, ymin:ymax, :]
    # normalize
    
    img = normalize(img, mean=mean, std=std)
    
    return img


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    img = img.astype(np.float32) / 255.  # converting array of ints to floats
    img_a = img[:, :, 0]
    img_b = img[:, :, 1]
    img_c = img[:, :, 2]

    # Extracting single channels from 3 channel image
    # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)

    # normalizing per channel data:
    img_a = (img_a - mean[0]) / std[0]
    img_b = (img_b - mean[1]) / std[1]
    img_c = (img_c - mean[2]) / std[2]

    # putting the 3 channels back together:
    x1 = [[[], [], []]]
    x1[0][0] = img_a
    x1[0][1] = img_b
    x1[0][2] = img_c

    # x1 = mx.nd.array(np.asarray(x1))
    x1 = np.asarray(x1)
    
    return x1    

def softmax(logits):
    ps = np.exp(logits)
    ps /= np.sum(ps)
    
    return ps

class image_classifier(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("model", "classifier", "Model for classifying images", "Resnet-Classifier"),
                    ("int", "batch_size", "Model batch size", 1),
                ),
            "inputs":
                (
                    ("media[]", "video_in", "Camera input stream"),
                ),
            "outputs":
                (
                    ("media[video_in]", "video_out", "Camera output stream"),
                )
        }


    def init(self, parameters, inputs, outputs):
        try:
            self.frame_num = 0
            # Load model from the specified directory.
            print("loading the model...")
            self.model = panoramasdk.model()
            self.model.open(parameters.classifier, 1)
            print("model loaded")

            # Create input and output arrays.
            prob_info = self.model.get_output(0)
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.classes = get_classes()

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    
    def topk(self, array, k=5):
        enum_vals = [(i, val) for i, val in enumerate(array)]
        sorted_vals = sorted(enum_vals, key=lambda tup: tup[1])
        top_k = sorted_vals[::-1][:k]
        return [tup[0] for tup in top_k]

    
    def entry(self, inputs, outputs):

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            person_image = stream.image

            print("Processing Image...")
            x1 = preprocess(person_image)
            print("Processed Image")

            # Do inference on the new frame.
            print("Performing Detector Inference")
            self.model.batch(0, x1)
            self.model.flush()
            print("Inference Completed.")

            # Get the results.
            resultBatchSet = self.model.get_result()
            prob_batch = resultBatchSet.get(0)
            prob_batch.get(0, self.prob_array)
            logits = self.prob_array[0]
            
            topK = 5
            probs = softmax(logits)
            ind = self.topk(probs, k=topK)
            
            lines = ['The input picture is classified to be:']
            for j in range(topK):
                lines.append('class [%s], with probability %.3f.'% (self.classes[ind[j]], probs[ind[j]]))
            message = "\n".join(lines)
            stream.add_label(message, 0.25, 0.25)

        self.model.release_result(resultBatchSet)
        outputs.video_out[i] = stream

        return True


def main():
    image_classifier().run()


main()