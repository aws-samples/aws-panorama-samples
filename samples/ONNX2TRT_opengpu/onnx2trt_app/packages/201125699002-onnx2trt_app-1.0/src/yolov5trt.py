from collections import defaultdict
import pycuda.autoinit
import torch
import pycuda.driver as cuda
import tensorrt as trt
import logging
import time
import numpy as np
import utils

log = logging.getLogger('my_logger')

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, batch_size, dynamic=False, num_classes = 80):
        # Create a Context on this device,
        log.info('Loading CFX')
        self.cfx = cuda.Device(0).make_context()
        log.info('Loaded CFX')
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
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
        self.time_buffers = defaultdict(list)

    def infer(self, input_image_batch):
        self.cfx.push()
        # Make self the active context, pushing it on top of the context stack.
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        t1 = time.time()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image_batch.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
        # Run inference.
        t2 = time.time()
        context.execute_v2(bindings=bindings)
        t3 = time.time()
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])
        
        # # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output_batch = host_outputs[0]
        t4 = time.time()
        # Do postprocess
        self.time_buffers["memcp"].append(t2-t1)
        self.time_buffers["infer"].append(t3-t2)
        self.time_buffers["post_memcp"].append(t4-t3)
        self.time_buffers["total"].append(t4-t1)
        
        for key, val_list in self.time_buffers.items():
            if(len(val_list)>=500//self.batch_size):
                timespent_ms = np.mean(val_list)
                log.info('Time Spent {}: {}ms'.format(key,timespent_ms))
                self.time_buffers[key] = []
        
        return output_batch

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image_batch(self, image_list):
        """
        Preprocess a list of image. We preprocess each image individually to prevent the case
        that each image has different input dimension. (Cannot batch before preprocessing)
        By default the image will be resized to 640x640.
        
        param:
            image_list: a list of np int8 array.

        return:
            A batch of float32 image. Shape [B, C, H, W]
        """
        return np.vstack([utils.preprocess(image) for image in image_list])
    
    def post_process_batch(self, pred, preprocessed_image, orig_image, filtered_classes = None, conf_thres=0.5, iou_thres=0.45):
        pred = np.reshape(pred, (self.batch_size,-1, self.num_classes+5)) # [x,y,w,h, object_score] + [coco 80 scores] => 85
        pred = torch.from_numpy(pred)
        pred = utils.non_max_suppression(pred, conf_thres = conf_thres, 
            iou_thres=iou_thres, classes=filtered_classes)
        output = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = utils.scale_coords(preprocessed_image.shape[1:], det[:, :4], orig_image.shape).round()
                output.append(det)
        return output