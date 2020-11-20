from __future__ import division
from __future__ import print_function

import panoramasdk
import cv2
import numpy as np
import boto3

# Global Variables

HEIGHT = 512
WIDTH = 512

class PikachuDetection(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "threshold", "Detection threshold", 0.50),
                    ("model", "pokemon_detection", "Model for detecting pokemon", "pikachu-detection-hyb"),
                    ("int", "batch_size", "Model batch size", 1),
                    ("float", "pokemon_index", "pokemon index based on dataset used", 0),
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
            # Frame Number Initialization
            self.frame_num = 0
            # Index for pokemon from parameters
            self.index = parameters.pokemon_index
            # Set threshold for model from parameters 
            self.threshold = parameters.threshold
            # set number of pokemon
            self.number_pokemon = 0
            
            # Load model from the specified directory.
            print("loading the model...")
            self.model = panoramasdk.model()
            self.model.open(parameters.pokemon_detection, 1)
            print("model loaded")

            # Create input and output arrays.
            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(rect_info.get_dims(), dtype=rect_info.get_type())

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def preprocess(self, img):
        resized = cv2.resize(img, (HEIGHT, WIDTH))

        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]  # RGB

        img = resized.astype(np.float32) / 255.  # converting array of ints to floats
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
    
    def get_number_pokemon(self, class_data, prob_data):
        # get indices of people detections in class data
        pokemon_indices = [i for i in range(len(class_data)) if int(class_data[i]) == self.index]
        # use these indices to filter out anything that is less than 95% threshold from prob_data
        prob_pokemon_indices = [i for i in pokemon_indices if prob_data[i] >= self.threshold]
        return prob_pokemon_indices

    def entry(self, inputs, outputs):
        self.frame_num += 1

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            pokemon_image = stream.image

            stream.add_label('Number of Pikachu : {}'.format(self.number_pokemon), 0.6, 0.05)

            x1 = self.preprocess(pokemon_image)
            
            # Do inference on the new frame.
            self.model.batch(0, x1)
            self.model.flush()

            # Get the results.
            resultBatchSet = self.model.get_result()

            class_batch = resultBatchSet.get(0)
            prob_batch = resultBatchSet.get(1)
            rect_batch = resultBatchSet.get(2)

            class_batch.get(0, self.class_array)
            prob_batch.get(0, self.prob_array)
            rect_batch.get(0, self.rect_array)

            class_data = self.class_array[0]
            prob_data = self.prob_array[0]
            rect_data = self.rect_array[0]
            
            
            # Get Indices of classes that correspond to Pokemon
            pokemon_indices = self.get_number_pokemon(class_data, prob_data)
            
            print('pokemon indices is {}'.format(pokemon_indices))
            
            try:
                self.number_pokemon = len(pokemon_indices)
            except:
                self.number_pokemon = 0
                
            
            # Draw Bounding Boxes on HDMI Output
            if self.number_pokemon > 0:
                for index in pokemon_indices:
    
                    left = np.clip(rect_data[index][0] / np.float(HEIGHT), 0, 1)
                    top = np.clip(rect_data[index][1] / np.float(HEIGHT), 0, 1)
                    right = np.clip(rect_data[index][2] / np.float(HEIGHT), 0, 1)
                    bottom = np.clip(rect_data[index][3] / np.float(HEIGHT), 0, 1)
    
                    stream.add_rect(left, top, right, bottom)
                    stream.add_label(str(prob_data[index][0]), right, bottom) 

            
            stream.add_label('Number of Pikachu : {}'.format(self.number_pokemon), 0.6, 0.05)
            
            self.model.release_result(resultBatchSet)
            outputs.video_out[i] = stream

        return True


def main():
    PikachuDetection().run()


main()
