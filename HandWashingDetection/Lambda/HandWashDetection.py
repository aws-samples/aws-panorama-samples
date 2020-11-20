import panoramasdk
import cv2
import numpy as np
import time
from classes import load_classes
import boto3
import json

s3 = boto3.resource('s3')
iot = boto3.client('iot-data', region_name='us-east-1')

class HandWashDetection(panoramasdk.base):
    def interface(self):
        return {
            "parameters":
            (
                ("model", "action_detection", "Model for detecting_action", "action-kinetics"),
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
            # Frame Number
            self.frame_num = 0

            # Load model from the specified directory
            print("loading model")
            self.action_detection_model = panoramasdk.model()
            self.action_detection_model.open(parameters.action_detection, 1)
            print("model loaded")

            # panorama SDK specific declarations
            self.class_name_list = []
            self.class_prob_list = []
            class_info = self.action_detection_model.get_output(0)
            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            
            # Set up Timer
            self.time_start = time.time()
            self.seconds_to_wash = 120
            
            # Misc
            self.list_actions = []
            self.message = ''

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def preprocess(self, img, size):
        resized = cv2.resize(img, (size, size))
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
        

    def entry(self, inputs, outputs):
        
        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            
            # Pre Process Frame
            prep_frame = self.preprocess(stream.image, 224)

            # Predict
            self.action_detection_model.batch(0, prep_frame)
            self.action_detection_model.flush()

            # Get the results.
            resultBatchSet = self.action_detection_model.get_result()
            
            class_batch = resultBatchSet.get(0)
            class_batch.get(0, self.class_array)
            class_data = self.class_array[0]
            
            # Load Classes
            classes = load_classes()
            
            # declare topKlist
            topKlist = []
            
            # Collect the Top 10 Classes
            sorted_vals = sorted(((value,index) for index, value in enumerate(class_data)), reverse=True)
            ind = [d for (c,d) in sorted_vals][0:10]
            for z in range(len(ind)):
                class_name = classes[ind[z]]
                topKlist.append(class_name)
                        
            # Here we want to be certain that we are detecting washing hands
            if 'washing_hands' in topKlist:
                self.list_actions.append(1)
            else:
                self.list_actions.append(0)
            
            if len(self.list_actions) > 20:
                self.list_actions = self.list_actions[-20:]
                    
            # Once we definitively detect washing hands, we start writing to the frame
            if ('washing_hands' in topKlist and sum(self.list_actions) > 10):
                hours, rem = divmod(time.time() - self.time_start, 3600)
                minutes, seconds = divmod(rem, 60)
                count_seconds = self.seconds_to_wash - int(minutes*60 + seconds)
                
                # write on frame 
                stream.add_label('Action: Washing Hands',0.1,0.25)
                
                if count_seconds <= 0:
                    stream.add_label('Time : {}'.format(0),0.1,0.5)
                    stream.add_label('Message : {}'.format('Great Job'),0.1,0.75)
                    self.message = 'Great Job'
                    
                # Reset Time start after 10 seconds. Until then show message
                if count_seconds <= -10:
                    stream.add_label('Time : {}'.format(0),0.1,0.5)
                    self.time_start = time.time()
                    self.message = ''
                else:
                    stream.add_label('Time : {}'.format(count_seconds),0.1,0.5)
                    self.message = ''
            
            # Top k action list to MQTT message
            response = iot.publish(topic='HandWashing',qos=1,
                                   payload=json.dumps({"results":topKlist,"message":self.message}))
        
            self.action_detection_model.release_result(resultBatchSet)
            outputs.video_out[i] = stream

        return True

def main():
    HandWashDetection().run()

main()
