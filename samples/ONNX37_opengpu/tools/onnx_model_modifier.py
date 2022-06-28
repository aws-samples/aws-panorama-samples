import onnx
from onnxmltools.utils import float16_converter

def change_input_dim(model):
    # "N" means var input dim, integer nums means actual batch size
    sym_batch_dim="batch"

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        if type(sym_batch_dim)==str:
            # update dim to be a symbolic value
            dim1.dim_param = sym_batch_dim
        elif type(sym_batch_dim)==int:
            # or update it to be an actual value:
            dim1.dim_value = sym_batch_dim



if __name__ == '__main__':
    # example usage here
    # convert to dynamic batch size
    model = onnx.load("batch8_yolov5s.onnx")
    change_input_dim(model)
    onnx.save(model, "batch_dynamic_yolov5s.onnx")

    model = onnx.load("batch_dynamic_yolov5s.onnx")
    fp16_model = float16_converter.convert_float_to_float16(model,keep_io_types=True)
    onnx.save(fp16_model, "batch_dynamic_fp16_yolov5s.onnx")