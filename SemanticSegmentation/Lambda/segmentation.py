from __future__ import division
from __future__ import print_function


import panoramasdk
import cv2
import numpy as np
import boto3


def plot_mask(img, predict, alpha=0.5):
    classes = np.unique(predict)
    w, h, _ = img.shape
    print(img.shape)
    for i, cla in enumerate(classes):
        try:
            color = np.random.random(3) * 255
            mask = np.repeat((predict == cla)[:, :, np.newaxis], repeats=3, axis=2).astype('uint8')
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask * 255
            mask = cv2.resize(mask, (h, w))
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(img, contours, -1, color, 5)
        except E:
            print(E)
    return img
    
def preprocess(img, shape=(512, 512)):
    
    resized = cv2.resize(img, shape) # (h, w)
    x1 = normalize(resized)
    return x1, resized
        
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

class segmentation(panoramasdk.base):
    
    def interface(self):
        return {
            "parameters":
                (
                    ("model", "segmodel", "Model for people detecting", "segmentation-model2"),
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
            self.model.open(parameters.segmodel, 1)
            print("model loaded")
            
            # Create input and output arrays.
            mask_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
        
            self.mask_array = np.empty(mask_info.get_dims(), dtype=mask_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            
            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False
        
    def entry(self, inputs, outputs):

        for i in range(len(inputs.video_in)):

            stream = inputs.video_in[i]
            
            person_image = stream.image
            
            #x1 = self.preprocess(person_image)
            print("Processing Image...")
            x1, orig_img = preprocess(person_image, (400, 500))
            print("Processed Image")
            
            
            # Do inference on the new frame.
            print("Performing Detector Inference")
            self.model.batch(0, x1)
            self.model.flush()
            print("Inference Completed.")

            # Get the results.
            resultBatchSet = self.model.get_result()

            mask_batch = resultBatchSet.get(0)
            prob_batch = resultBatchSet.get(1)
            
            mask_batch.get(0, self.mask_array)
            prob_batch.get(0, self.prob_array)
        
            masks = np.squeeze(np.argmax(self.mask_array, 1))
            _ = plot_mask(stream.image, masks)

        
            self.model.release_result(resultBatchSet)
            outputs.video_out[i] = stream

        return True


def main():
    segmentation().run()


main()