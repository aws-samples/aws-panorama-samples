import time

import panoramasdk
import cv2
import numpy as np

import utils

INPUT_SIZE = 640
MODEL = 'yolov5s-640'
THRESHOLD = 0.5
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class YOLOv5(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "threshold", "Detection threshold", THRESHOLD),
                    ("model", "model", "YOLOv5 pre-trained model (MS COCO)", MODEL),
                    ("int", "input_size", "Model input size (actual shape will be [1, 3, input_size, input_size])", INPUT_SIZE)
                ),
            "inputs":(("media[]", "video_in", "Camera input stream"),),
            "outputs":(("media[video_in]", "video_out", "Camera output stream"),)
        }

    def log(self, msg, frequency=1000):
        """Log the messages at a reduced rate, can reduce AWS CloudWatch logs polluting and simplify log searching
        params:
            msg (str): message to log
            frequency (int): message will be logged only once for this number of processed image frames
        """ 
        if self.frame_num % frequency == 1:
            print(f'[Frame: {self.frame_num}]: {msg}')
    
    def run_inference(self, image):
        self.log("Running inference")
        start = time.time()
        self.model.batch(0, self.processor.preprocess(image))
        self.model.flush()
        result = self.model.get_result()
        inf_time = time.time() - start
        self.log(f'Inference completed in {int(inf_time * 1000):,} msec', frequency=1000)

        batch_0 = result.get(0)
        batch_1 = result.get(1)
        batch_2 = result.get(2)
        batch_0.get(0, self.pred_0)
        batch_1.get(0, self.pred_1)
        batch_2.get(0, self.pred_2)
        self.model.release_result(result)
        
        return inf_time
        
    def init(self, parameters, inputs, outputs):
        print('init()')
        try:
            self.input_size = parameters.input_size
            self.class_names = CLASSES
            self.processor = utils.Processor(self.class_names, self.input_size, keep_ratio=True)
            self.threshold = parameters.threshold
            self.frame_num = 0
            
            # Load model from the specified directory.
            print(f'Loading model {parameters.model}')
            start = time.time()
            self.model = panoramasdk.model()
            self.model.open(parameters.model, 1)
            print(f'Model loaded in {int(time.time() - start)} seconds')

            # Create output arrays
            info_0 = self.model.get_output(0)
            info_1 = self.model.get_output(1)
            info_2 = self.model.get_output(2)

            self.pred_0 = np.empty(info_0.get_dims(), dtype=info_0.get_type())
            self.pred_1 = np.empty(info_1.get_dims(), dtype=info_1.get_type())
            self.pred_2 = np.empty(info_2.get_dims(), dtype=info_2.get_type())
            
            # Use all the model outputs
            self.predictions = [self.pred_0, self.pred_1, self.pred_2]

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False
            
    def entry(self, inputs, outputs):
        try:
            self.frame_num += 1
            self.log('entry()')
            for i in range(len(inputs.video_in)):
                stream = inputs.video_in[i]
                in_image = stream.image

                inf_time = self.run_inference(in_image)
                
                start = time.time()
                (boxes, scores, cids), _ = self.processor.post_process(self.predictions, in_image.shape, self.threshold)
                post_time = time.time() - start

                for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cids):
                    label = f'{self.class_names[int(cid)]} {score:.2f}'
                    stream.add_rect(x1, y1, x2, y2)
                    stream.add_label(label, x1, y1)  # (x, y) here are coords of top-left label location
            
            # Visual log of frames and processing times, remove if not needed
            msg = f'Frame: {self.frame_num:,}\n- infer: {int(inf_time * 1000)} msec\n- post-proc: {int(post_time * 1000)} msec'
            stream.add_label(msg, 0.6, 0.1)
            
            outputs.video_out[i] = stream
            
        except Exception as e:
            print("Exception: {}".format(e))
            return False
        
        return True     


def main():
    YOLOv5().run()


main()
