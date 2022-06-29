import tensorrt as trt
# source from https://github.com/bei91/yolov5-onnx-tensorrt/blob/master/demo/onnx_tensorrt.py
# dynamic batch source from https://github.com/egbertYeah/simple_tensorrt_dynamic/blob/main/onnx-tensorrt.py

def onnx2tensorrt(onnx_path, output_path, fp=16, dynamic_batch = [1, 4, 8], img_height=640, img_width=640, max_workspace_GB = 1):
    """
    Converts Onnx model to tensorrt engine.
    
    Args:
        onnx_path: string. the path to onnx model. 
        output_path: string. the output path and filename of the engine.
        fp: int. 16 or 32. Engine file floating point.
        dynamic_batch: list of one or three elements.
            List of three elements. Supports dynamic batch size. ex: [1, 4, 8] means the min batch, optimal batch, max batch.
            List of one element. ex: [4] means the static batch size is 4. 
        img_height: int. image height.
        img_width: int. image width.
    """

    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # trt7
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
        
        # Parsing ONNX File
        with open(onnx_path, 'rb') as f:
            print('Beginning ONNX file parsing', flush=True)
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("ERROR", parser.get_error(error))
        print("num layers:", network.num_layers, flush=True)

        # Engine file builder config
        static_batch = len(dynamic_batch)==1
            
        builder.max_batch_size = dynamic_batch[-1]
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_GB* (1 << 30) # 8GB for TRT to use.
        
        ## Seting FP16 or 32
        if fp == 16:
            config.set_flag(trt.BuilderFlag.FP16)
            # builder.fp16_mode = True
        elif fp != 32:
            raise ValueError("fp {} not supported".format(fp))
        
        ## Setting Real Shape, -1 as dynamic
        network.get_input(0).shape = [ dynamic_batch[-1] , 3, img_height, img_width] if static_batch else [-1, 3, img_height, img_width]
        # network.get_input(0).shape = [dynamic_batch[-1], 3, img_height, img_width]  # trt7, -1 as dynamic
        ## Using Dynamic Shape or Not
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/OptimizationProfile.html#tensorrt.IOptimizationProfile
        dynamic_shapes = {}
        if not static_batch:
            shapes = [ (batch, 3, img_height, img_width) for batch in dynamic_batch]
            dynamic_shapes={ network.get_input(0).name: shapes }
            print("Using Dynamic Shape", dynamic_shapes)
            profile = builder.create_optimization_profile()

            for binding_name, dynamic_shape in dynamic_shapes.items():
                min_shape, opt_shape, max_shape = dynamic_shape
                profile.set_shape(binding_name, min_shape, opt_shape, max_shape)

            config.add_optimization_profile(profile)

        print("Start compiling to engine file, it takes 8-12 mins ...", flush=True)
        engine = builder.build_engine(network, config)
        print("Engine built, start serializing ...", flush=True)
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        print("Completed creating Engine", flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to tensorrt engine.')
    parser.add_argument('-b','--batchsize', nargs='+', help='Setting the converted batch size', required=True, default=[1,4,8])
    parser.add_argument('-p','--fp', type=int, help='Floating Point Precision', required=True, default=16)
    parser.add_argument('-i','--input_path', type=str, help='Input. The onnx model path', required=True)
    parser.add_argument('-o','--output_path', type=str, help='Output. The engine path', required=True)
    args = parser.parse_args()
    args.batchsize = [ int(a) for a in args.batchsize]
    onnx2tensorrt(args.input_path, args.output_path, fp=args.fp, dynamic_batch=args.batchsize)
