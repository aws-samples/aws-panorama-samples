import panoramasdk
import cv2
import numpy as np
import time
from config import CLASSES

class ActionDetection(panoramasdk.base):
    def interface(self):
        return {
            "parameters":
            (
                ("float", "threshold",      "Detection threshold",        0.05),
                ("model", "action_detection", "Model for detecting_action", "action-kinetics"),
                ("int",   "batch_size",     "Model batch size",           1),
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
            self.class_name_list = []
            self.class_prob_list = []

            # Detection probability threshold.
            self.threshold = parameters.threshold

            # Load model from the specified directory.
            print("loading model")
            self.model = panoramasdk.model()
            self.model.open(parameters.action_detection, 1)
            print("model loaded")

            class_info = self.model.get_output(0)
            
            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def preprocess(self, img):
        resized = cv2.resize(img, (224, 224))

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

        x1 = np.asarray(x1)
        return x1
        
    def update_classes(self, class_list):
        new_class_list = []
        ## function desgined to merge other classes apart from smoking into not_smoking.
        for item in class_list:
            if item not in ('smoking','smoking_hookah'):
                new_class_list.append('not_smoking')
            else:
                new_class_list.append('smoking')
        return new_class_list

    def entry(self, inputs, outputs):
        self.frame_num += 1

        for i in range(len(inputs.video_in)):
            start_time = time.time()
            stream = inputs.video_in[i]
            x1 = self.preprocess(stream.image)

            # Do inference on the new frame.
            self.model.batch(0, x1)
            self.model.flush()

            # Get the results.
            resultBatchSet = self.model.get_result()
            class_batch = resultBatchSet.get(0)
            class_batch.get(0, self.class_array)
            class_data = self.class_array[0]
            
            self.model.release_result(resultBatchSet)
            stream.add_label('Probability :',0.6, 0.50)
            stream.add_label('***********************',0.6, 0.55)
            updated_classes = self.update_classes(CLASSES)            
            output_dict = {}
            for item in set(updated_classes):
                output_dict[item] = 0.0
            print("output_dict is {}", output_dict)
            
            num_classes = len(CLASSES)
            for p in range(num_classes):
                key = updated_classes[p]
                val = class_data[i]
                output_dict[key] += val
                
            print("output_dict is {}", output_dict)
            text_pos =  0.55
            tot = sum(output_dict.values())
            for q in set(updated_classes):
                text_pos += 0.05
                disp_res = output_dict[q] / tot
                stream.add_label('{} : {}'.format(q, np.round(disp_res, 3)), 0.6, text_pos)
            
            outputs.video_out[i] = stream

        return True

def main():
    ActionDetection().run()

main()
