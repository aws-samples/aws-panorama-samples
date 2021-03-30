import panoramasdk
import cv2
import numpy as np
import boto3
from mjpeg_server import PanoramaMJPEGServer

HEIGHT = 512
WIDTH = 512

class people_counter(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "threshold", "Minimum confidence for display", 0.10),
                    ("model", "people_counter", "Name of the model in AWS Panorama", "aws-panorama-sample-model"),
                    ("int", "batch_size", "Model batch size", 1),
                    ("float", "person_index", "The index of the person class in the model's dataset", 14),
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
            self.threshold = parameters.threshold
            self.person_index = parameters.person_index
            self.threshold = parameters.threshold
            self.frame_num = 0
            self.number_people = 0
            self.colours = np.random.rand(32, 3)

            print("Loading model: " + parameters.people_counter)
            self.model = panoramasdk.model()
            self.model.open(parameters.people_counter, 1)

            print("Creating input and output arrays")
            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(rect_info.get_dims(), dtype=rect_info.get_type())
            self.mjpegserver = PanoramaMJPEGServer()

            print("Initialization complete")
            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def preprocess(self, img):
        resized = cv2.resize(img, (HEIGHT, WIDTH))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img = resized.astype(np.float32) / 255.
        img_a = img[:, :, 0]
        img_b = img[:, :, 1]
        img_c = img[:, :, 2]

        # Normalize data in each channel
        img_a = (img_a - mean[0]) / std[0]
        img_b = (img_b - mean[1]) / std[1]
        img_c = (img_c - mean[2]) / std[2]

        # Put the channels back together
        x1 = [[[], [], []]]
        x1[0][0] = img_a
        x1[0][1] = img_b
        x1[0][2] = img_c

        x1 = np.asarray(x1)
        return x1

    def get_number_persons(self, class_data, prob_data):
        # Filter out results beneath confidence threshold
        person_indices = [i for i in range(len(class_data)) if int(class_data[i]) == self.person_index]
        prob_person_indices = [i for i in person_indices if prob_data[i] >= self.threshold]
        return prob_person_indices

    def entry(self, inputs, outputs):
        self.frame_num += 1

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            person_image = stream.image
            stream.add_label('People detected: {}'.format(self.number_people), 0.8, 0.05)

            # Prepare the image and run inference
            x1 = self.preprocess(person_image)
            self.model.batch(0, x1)
            self.model.flush()
            resultBatchSet = self.model.get_result()

            # Process results
            class_batch = resultBatchSet.get(0)
            prob_batch = resultBatchSet.get(1)
            rect_batch = resultBatchSet.get(2)

            class_batch.get(0, self.class_array)
            prob_batch.get(0, self.prob_array)
            rect_batch.get(0, self.rect_array)

            class_data = self.class_array[0]
            prob_data = self.prob_array[0]
            rect_data = self.rect_array[0]

            # Get indices of people classes
            person_indices = self.get_number_persons(class_data,prob_data)

            try:
                self.number_people = len(person_indices)
            except:
                self.number_people = 0

            # Draw bounding boxes on output image
            if self.number_people > 0:
                for index in person_indices:

                    left = np.clip(rect_data[index][0] / np.float(HEIGHT), 0, 1)
                    top = np.clip(rect_data[index][1] / np.float(WIDTH), 0, 1)
                    right = np.clip(rect_data[index][2] / np.float(HEIGHT), 0, 1)
                    bottom = np.clip(rect_data[index][3] / np.float(WIDTH), 0, 1)

                    stream.add_rect(left, top, right, bottom)
                    stream.add_label(str(prob_data[index][0]), right, bottom)
            # Add text
            stream.add_label('People detected: {}'.format(self.number_people), 0.8, 0.05)

            # Feed into mjpeg server
            self.mjpegserver.feed_frame(stream.image)

            self.model.release_result(resultBatchSet)
            outputs.video_out[i] = stream

        return True


def main():
    people_counter().run()

main()
