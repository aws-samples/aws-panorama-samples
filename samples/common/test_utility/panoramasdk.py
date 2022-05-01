import sys
import os
import time
import json
import logging
from logging.handlers import RotatingFileHandler

import IPython
import matplotlib.pyplot as plt
import cv2
import dlr

import panorama_test_utility_graph

# -------

# configure DLR
dlr.counter.phone_home.PhoneHome.disable_feature()

# configure logging
logging.basicConfig(filename="SimulatorLog.log")
log_p = logging.getLogger('panoramasdk')
log_p.setLevel(logging.DEBUG)
handler = RotatingFileHandler("SimulatorLog.log", maxBytes=100000000, backupCount=2)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
log_p.addHandler(handler)

# configure matplotlib
plt.rcParams["figure.figsize"] = (20, 20)

# -------
# Globals

_c = None

# -------
# Custom exceptions

class TestUtilityBaseError(Exception):
    pass

class TestUtilityEndOfVideo(TestUtilityBaseError):
    pass

# -------

def _configure( config ):
    
    global _c
    
    _c = config


class media(object):

    """
    This class mimics the Media object in Panoroma SDK .

    ...

    Attributes
    ----------
    inputpath : str
        input path is a string path to a video file

    gen : video_reader instance

    Methods
    -------
    video_reader(key)
        CV2 based Generator for Video

    image : Property
        Gets the next frame using the generator

    image : Property Setter
        If we want to set the value of stream.image

    add_label:
        Add text to frame

    add_rect:
        add rectangles to frame

    time_stamp:
        returns the current timestamp

    stream_uri:
        returns a hardcoded value for now

    """

    def __init__(self, array):
        self.image = array
        self.w, self.h, self.c = array.shape

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, array):
        self.__image = array

    def add_label(self, text, x1, y1):

        if x1 > 1 or y1 > 1:
            raise ValueError('Value should be between 0 and 1')

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (int(x1 * self.h), int(y1 * self.w))

        # fontScale
        fontScale = 1

        # White in BGR
        color = (255, 255, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        self.__image = cv2.putText(self.__image, text, org, font,
                                   fontScale, color, thickness, cv2.LINE_AA)

    def add_rect(self, x1, y1, x2, y2):

        if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
            raise ValueError('Value should be between 0 and 1')

        # Red in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        start_point = (int(x1 * self.h), int(y1 * self.w))
        end_point = (int(x2 * self.h), int(y2 * self.w))

        self.__image = cv2.rectangle(
            self.__image,
            start_point,
            end_point,
            color,
            thickness)

    @property
    def time_stamp(self):
        micro = 1000000
        t = int( time.time() * micro )
        return ( t // micro, t % micro )

    @property
    def stream_uri(self):
        """Hardcoded Value for Now"""

        return 'cam1'
    
    @property
    def stream_id(self):
        """Hardcoded Value for Now"""

        return 'cam1_id'
    
    @property
    def is_cached(self):
        """Hardcoded Value for Now"""

        return True

class Video_Array(object):

    """
    This class is implemented so we can use the opencv VideoCapture Method

    ...

    Attributes
    ----------
    inputpath :
        Input is a path to a video


    Methods
    -------
    get_frame
        returns a frame at a time until it exceeds _c.video_range
    """

    def __init__(self, inputpath):

        self.input_path = inputpath

        assert _c.video_range.start >= 0, "Config.video_range.start has to be >= 0."
        assert _c.video_range.stop > 0, "Config.video_range.stpp has to be positive integer."
        assert _c.video_range.step > 0, "Config.video_range.step has to be positive integer."

    def get_frame(self):
        
        if not os.path.exists(self.input_path):
            raise FileNotFoundError( self.input_path )
        
        cap = cv2.VideoCapture(self.input_path)
        frame_num = 0

        # Skip first frames based on video_range.start
        for i in range(0,_c.video_range.start):
            _, frame = cap.read()
            frame_num += 1
            
            if frame is None:
                return

        while (frame_num <= _c.video_range.stop):
            
            _, frame = cap.read()
            frame_num += 1

            if frame is None:
                return
            
            # Reading frame one by one to reduce memory space
            yield frame

            # Skip frames based on video_range.step
            for i in range(1,_c.video_range.step):
                _, frame = cap.read()
                frame_num += 1


class PortImpl:
    pass

class MediaSourceRtspCameraPort(PortImpl):
    def __init__(self):
        self.video_obj = Video_Array(_c.videoname).get_frame()

    def get(self):
        try:
            return [media(next(self.video_obj))]
        except StopIteration:
            raise TestUtilityEndOfVideo("Reached end of video")
    
class ParameterPort(PortImpl):
    def __init__( self, producer_node ):
        self.producer_node = producer_node

    def get(self):
        return self.producer_node.node_elm["value"]

    
class HdmiDataSinkPort(PortImpl):

    def __init__(self):
        self.screenshot_n_frame = 0

    def put( self, data ):
    
        media_list = data

        if _c.screenshot_dir:
            for i_media, media_obj in enumerate(media_list):
                filename = f"{_c.screenshot_dir}/screenshot_%d_%04d.png" % ( i_media, self.screenshot_n_frame )
                cv2.imwrite( filename, media_obj.image )
            
            self.screenshot_n_frame += 1
        
        if _c.render_output_image_with_pyplot:
            for media_obj in media_list:
                IPython.display.clear_output(wait=True)
                plt.imshow( cv2.cvtColor(media_obj.image,cv2.COLOR_BGR2RGB) )
                plt.show()
    
class port:
    
    def __init__( self, producer_node=None, consumer_node=None ):
    
        assert producer_node or consumer_node
        assert producer_node is None or consumer_node is None
    
        self.producer_node = producer_node
        self.consumer_node = consumer_node
    
        if self.producer_node:
            if isinstance( self.producer_node, panorama_test_utility_graph.MediaSourceRtspCameraNode ):
                self.impl = MediaSourceRtspCameraPort()
            elif isinstance( self.producer_node, panorama_test_utility_graph.ParameterNode ):
                self.impl = ParameterPort( self.producer_node )
            else:
                raise ValueError( "Unsupported producer node type", type(self.producer_node) )
        
        else:
            if isinstance( self.consumer_node, panorama_test_utility_graph.HdmiDataSinkNode):
                self.impl = HdmiDataSinkPort()
            else:
                raise ValueError( "Unsupported consumer node type", type(self.consumer_node) )
    
    def get(self):
        
        if not self.producer_node:
            raise ValueError( "port.get is supported only by input ports" )
            
        return self.impl.get()

    def put( self, data ):

        if not self.consumer_node:
            raise ValueError( "port.put is supported only by output ports" )
        
        return self.impl.put(data)
    

class node(object):
    """
    This class is implemented to mimic Panoroma Node Class.

    ...

    Attributes
    ----------

    Methods
    -------
    """
    
    # Add properties and methods to the instance
    @staticmethod
    def _initialize(instance):
        
        graph = panorama_test_utility_graph.Graph()
        graph.load(
            app_dir_top = f"./{_c.app_name}",
            app_name = _c.app_name,
        )
        
        class Ports:
            pass

        instance.inputs = Ports()
        for name, producer_node in graph.business_logic_node.inputs.items():
            print( "Initializing input port", name, producer_node )
            setattr( instance.inputs, name, port(producer_node = producer_node) )
        
        instance.outputs = Ports()
        for name, consumer_node in graph.business_logic_node.outputs.items():
            print( "Initializing output port", name, consumer_node )
            setattr( instance.outputs, name, port(consumer_node = consumer_node) )


    # Create node instance
    # This method is automatically called even if it is not called explicitly
    def __new__(cls, *args, **kwargs):

        instance = super(node,cls).__new__(cls, *args, **kwargs)

        node._initialize( instance )

        node._dlr_models = {}

        return instance

    # Instantiate DLRModel when it is used for the first time, 
    # and check if the model node/interface are correctly defined in JSON files
    def _load_dlr_model( self, name ):
        
        # Instantiate DLRModel
        model_path = _c.models[ name ]  + "-" + _c.compiled_model_suffix
        model = dlr.DLRModel( model_path )
        self._dlr_models[name] = model

    def call( self, input, name, time_out = None ):

        if name not in self._dlr_models:
            self._load_dlr_model(name)

        assert name in self._dlr_models

        dlr_model = self._dlr_models[ name ]
        output = dlr_model.run( input )

        assert isinstance( output, list ), f"Unexpected output type {type(output)}"

        return tuple(output)

