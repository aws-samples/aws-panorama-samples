import pycuda.autoinit
import torch
import torchvision
import pycuda.driver as cuda
import tensorrt as trt
import logging
import time
import cv2
import numpy as np
import random
import os

log = logging.getLogger('my_logger')

# load coco labels
categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

HUMAN_CLASS  = categories.index("person")

    

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )



class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, batch_size, dynamic=False, num_classes = 80, input_w=640, input_h=640, 
                 conf_threshold = 0.25, iou_threshold = 0.4):
        # Create a Context on this device,
        log.info('Loading CFX')
        self.cfx = cuda.Device(0).make_context()
        log.info('Loaded CFX')
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        self.input_w = input_w
        self.input_h = input_h
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dynamic=dynamic
        # Deserialize the engine from file
        
        log.info('Loading Engine File')
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        log.info('Loaded Engine File')

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        # assert self.batch_size == engine.max_batch_size
        for binding in engine:
            # not sure why trt takes -1 into account
            size = None
            if self.dynamic:
                size = abs(trt.volume(engine.get_binding_shape(binding))*self.batch_size)
            else:
                size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        
        context.set_binding_shape(0, (self.batch_size, 3, self.input_h, input_w))
        
        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.duration_buffer = []
    
    def infer(self, image_raw_list):
        # # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image_list, image_raw_list, origin_h_list, origin_w_list = \
            self.preprocess_image_list(image_raw_list)
        input_image_batch = np.stack(input_image_list, axis=0)
        start=time.time()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image_batch.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output_batch = host_outputs[0]
        duration=(time.time()-start)
        self.duration_buffer.append(duration)
        if(len(self.duration_buffer)>=100//self.batch_size):
            buf = np.array(self.duration_buffer)
            fps = 1.0/np.mean(buf)*self.batch_size
            log.info('FPS: {}'.format(fps))
            self.duration_buffer = []
        # Do postprocess
        result_boxes_list, result_scores_list, result_classid_list = self.post_process_batch(
            output_batch, origin_h_list, origin_w_list
        )
        
        # Draw rectangles and labels on the original image
        for idx in range(len(result_boxes_list)):
            result_boxes = result_boxes_list[idx]
            result_scores = result_scores_list[idx]
            result_classid = result_classid_list[idx]
            image_raw = image_raw_list[idx]
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                plot_one_box(
                    box,
                    image_raw,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[i])], result_scores[i]
                    ),
                )

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image_list(self, image_list):
        """
        Preprocess a list of image. We preprocess each image individually to prevent the case
        that each image has different input dimension. (Cannot batch before preprocessing)
        
        param:
            image_list: a list of np int8 array.

        return:
            image_list: a list of preprocessed images
            image_raw_list: a list of original images
            h_list: a list of original height
            w_list: a list of original width
        """
        res = [self.preprocess_image(image_raw) for image_raw in image_list]
        image_list, image_raw_list, h_list, w_list = [list(sub_res) for sub_res in zip(*res)]
        return image_list, image_raw_list, h_list, w_list

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            image_raw: the original image, np int8
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, _ = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w
    
    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        
        return y

    def post_process_batch(self, output_batch, origin_h_list, origin_w_list):
        """
        TODO: make it real batch. Currently postprocssing results individually.
        """
        preds = np.reshape(output_batch, (self.batch_size,-1, self.num_classes+5)) # [x,y,w,h, object_score] + [coco 80 scores] => 85
        pred_list = [np.squeeze(pred, axis=0) for pred in np.split(preds, self.batch_size, axis=0)]
        result_boxes_list = [] 
        result_scores_list = [] 
        result_classid_list = []

        for i in range(len(pred_list)):
            result_boxes, result_scores, result_classid = \
                self.post_process(pred_list[i], origin_h_list[i], origin_w_list[i])
            result_boxes_list.append(result_boxes)
            result_scores_list.append(result_scores)
            result_classid_list.append(result_classid)
        return result_boxes_list, result_scores_list, result_classid_list
    
    def post_process(self, pred, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            pred:     A tensor likes [num_boxes,cx,cy,w,h,conf, num_class confidence] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4:5]
        # Get the classid
        class_scores = pred[:, 5:]
        class_scores*=scores
        scores, classid = class_scores.max(1, keepdim=True)
        scores = torch.squeeze(scores)
        classid = torch.squeeze(classid)
        # Choose those boxes that score > CONF_THRESH
        si = scores > self.conf_threshold
        #si = torch.logical_and(scores > CONF_THRESH, classid == HUMAN_CLASS)
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]

        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.iou_threshold).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()

        pi = result_classid == HUMAN_CLASS # people index
        result_boxes = result_boxes[pi, :]
        result_scores = result_scores[pi]
        result_classid = result_classid[pi]

        return result_boxes, result_scores, result_classid