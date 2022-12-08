import cv2
from yolov5trt import YoLov5TRT
import os
import utils
import argparse
parser = argparse.ArgumentParser(description='inference with engine.')
parser.add_argument('-i','--engine', type=str, help='The engine filepath', required=True)
args = parser.parse_args()


categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

model_batch_size = 1
pre_processing_output_size = 640
engine_file_path = args.engine
fp = 16
engine_batch_size = "1 4 8"
is_dynamic = True

yolov5_wrapper = YoLov5TRT(engine_file_path, model_batch_size, is_dynamic, len(categories),
    pre_processing_output_size, pre_processing_output_size)


input_images_batch = [os.path.join("./samples", fn )for fn in os.listdir("./samples")]*model_batch_size
input_images_batch = [cv2.imread(pth) for pth in input_images_batch]
input_images_batch = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in input_images_batch]

org_image_list = input_images_batch[:model_batch_size]
yolov5_wrapper.preprocess_image_batch(org_image_list)
yolov5_wrapper.infer()
prediction = yolov5_wrapper.post_process_batch() 

for image_idx, det_results in enumerate(prediction):
    for box_idx, bbox in enumerate(det_results):
        bbox = bbox.tolist()
        coord = bbox[:4]
        score = bbox[4]
        class_id = bbox[5]
        utils.plot_one_box(coord, org_image_list[image_idx],
            label="{}:{:.2f}".format(categories[int(class_id)], score))
    cv2.imwrite('{}.jpg'.format(image_idx), cv2.cvtColor(org_image_list[image_idx], cv2.COLOR_RGB2BGR))

yolov5_wrapper.destroy()