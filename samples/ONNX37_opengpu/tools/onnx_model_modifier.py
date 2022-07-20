import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16


if __name__ == '__main__':
    # example usage here
    # convert to dynamic batch size
    model = onnx.load("./onnx_models/yolov5s.onnx")
    fp16_model = convert_float_to_float16(model,keep_io_types=True)
    onnx.save(fp16_model, "./onnx_models/yolov5s_fp16.onnx")
