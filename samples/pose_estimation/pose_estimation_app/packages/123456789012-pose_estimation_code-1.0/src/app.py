import sys
import time
import math
import cProfile

import cv2
import numpy as np
import panoramasdk

# ---

people_detection_input_resolution = (600,480)
pose_estimation_input_resolution = (192,256)
pose_estimation_output_resolution = (48,64)

line_color = (0,0,255)
line_thickness = 1
dot_color = (64,128,255)
dot_size = np.array( [2,2] )
box_color = (0,0,255)
box_thickness = 1

skip_people_detection = False
skip_pose_estimation = False

# ---

def trace( s ):
    print( s, flush=True )
    pass

class Application:

    def __init__( self, node, launcher=None ):
    
        self.node = node
        self.frame_num = 0
        self.profile_requested = False
        self.screenshot_requested_frames = 0
        self.screenshot_current_frame = 0
        
        if launcher:
            launcher.registerCommandHandler( "profile", self.command_Profile, in_main_thread = False )
            launcher.registerCommandHandler( "screenshot", self.command_Screenshot, in_main_thread = False )

    # run top-level loop of application  
    def run(self):
        
        while True:

            print( "frame : %d" % self.frame_num, flush=True )

            if self.profile_requested:
                self.profile_requested = False
                cProfile.runctx( "self.process_streams()", globals(), locals() )
            else:
                self.process_streams()

            self.frame_num += 1

    # process single frame for each camera input  
    def process_streams(self):
        
        trace("getting camera images 1")
        streams = self.node.inputs.video_in.get()
        trace("getting camera images 2")

        if self.frame_num==0:
            for i, stream in enumerate(streams):
                print( "streams[%d] : shape = %s" % ( i, stream.image.shape ), flush=True )

        for i_stream, stream in enumerate(streams):
            image = self.resize_and_normalize( stream.image )
            self.detect_people( i_stream, stream, image )

        if self.screenshot_current_frame < self.screenshot_requested_frames:
            for i_stream, stream in enumerate(streams):
                filename = "/opt/aws/panorama/storage/screenshot_%d_%04d.png" % ( i_stream, self.screenshot_current_frame )
                print( "Taking screenshot -", filename, flush=True )
                cv2.imwrite( filename, stream.image )
            self.screenshot_current_frame += 1

        trace("process_streams 8")
        self.node.outputs.video_out.put(streams)
        trace("process_streams 9")

    # Transform the camera image to model input format
    def resize_and_normalize( self, image ):

        resized = cv2.resize( image, people_detection_input_resolution )

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = resized.astype(np.float32) / 255.
        img_a = img[:, :, 0]
        img_b = img[:, :, 1]
        img_c = img[:, :, 2]

        # normalizing per channel data:
        img_a = (img_a - mean[0]) / std[0]
        img_b = (img_b - mean[1]) / std[1]
        img_c = (img_c - mean[2]) / std[2]
        
        # putting the 3 channels back together:
        x1 = [[[], [], []]]
        x1[0][0] = img_a
        x1[0][1] = img_b
        x1[0][2] = img_c
        
        return np.asarray(x1)

    # Detect people in a camera image, and call pose estimation
    def detect_people( self, i_stream, stream, image ):

        assert image.shape == (1,3,people_detection_input_resolution[1],people_detection_input_resolution[0])

        if not skip_people_detection:
        
            trace("detect_people 1")
            people_detection_results = self.node.call({"data":image}, "people_detection_model" )
            trace("detect_people 2")

            if people_detection_results is None:
                print( "people_detection_results is None", flush=True )
                return

            assert isinstance(people_detection_results,tuple) and len(people_detection_results)==3
        
            classes, scores, boxes = people_detection_results

            assert classes.shape == (1,100,1)
            assert scores.shape == (1,100,1)
            assert boxes.shape == (1,100,4)

            top4_index = np.argpartition( scores[0], -4, axis=None )[-4:]

        else:
            classes = np.array([[[0]]])
            scores = np.array([[[1]]])
            score = scores.item()
            boxes = np.array([[[ 0, 0, *people_detection_input_resolution ]]])
            top4_index = np.array( [0] )

        # filter out low score indeces
        top4_index_filtered = []
        for index in top4_index:
            if scores[0][index] > 0.2:
                top4_index_filtered.append(index)
        top4_index = np.array( top4_index_filtered )

        if top4_index.size == 0:
            return

        trace( "People detection scores : %s" % (scores[0][top4_index]) )

        pose_esimation_input = []

        for i_high_score in top4_index:

            box = boxes[0][i_high_score]

            box_in_camera_space = (
                int( box[0].item() * stream.image.shape[1] / people_detection_input_resolution[0] ),
                int( box[1].item() * stream.image.shape[0] / people_detection_input_resolution[1] ),
                int( box[2].item() * stream.image.shape[1] / people_detection_input_resolution[0] ),
                int( box[3].item() * stream.image.shape[0] / people_detection_input_resolution[1] ), 
            )
            
            cv2.rectangle( 
                stream.image, 
                box_in_camera_space[0:2], 
                box_in_camera_space[2:4], 
                color = box_color, thickness = box_thickness, lineType=cv2.LINE_8
            )

            x1, y1, x2, y2 = box
            x1 = math.floor( max( x1, 0 ) )
            y1 = math.floor( max( y1, 0 ) )
            x2 = math.ceil( min( x2, people_detection_input_resolution[0] ) )
            y2 = math.ceil( min( y2, people_detection_input_resolution[1] ) )
            box = np.array( [ x1, y1, x2, y2 ] )
            
            sub_image = image[ 0:1, :, y1:y2, x1:x2 ]
            sub_image_resized_r = cv2.resize( sub_image[0][0], pose_estimation_input_resolution )
            sub_image_resized_g = cv2.resize( sub_image[0][1], pose_estimation_input_resolution )
            sub_image_resized_b = cv2.resize( sub_image[0][2], pose_estimation_input_resolution )
            sub_image_resized = np.array( [ sub_image_resized_r, sub_image_resized_g, sub_image_resized_b ] )
            
            pose_esimation_input.append( sub_image_resized )

        self.estimate_pose( i_stream, stream, boxes[0][top4_index], np.array( pose_esimation_input ) )

    # Estimate poses of 1~4 people, and visualize the result
    def estimate_pose( self, i_stream, stream, boxes, images ):
    
        if skip_pose_estimation:
            return
        
        #print( "boxes.shape", boxes.shape, flush=True )
        #print( "images.shape", images.shape, flush=True )

        assert boxes.shape[0] <= 4
        assert boxes.shape[1:] <= (4,)
        
        assert images.shape[0] == boxes.shape[0]
        assert images.shape[1:] == (3,pose_estimation_input_resolution[1],pose_estimation_input_resolution[0])
        
        model_name = "pose_estimation_model_%d" % images.shape[0]

        trace("estimate_pose 1")
        estimate_pose_results = self.node.call({"data":images}, model_name )
        trace("estimate_pose 2")
        
        if estimate_pose_results is None:
            print( "estimate_pose_results is None", flush=True )
            return
        
        assert isinstance(estimate_pose_results,tuple) and len(estimate_pose_results)==1
        assert estimate_pose_results[0].shape[0] == boxes.shape[0]

        for pose, box in zip( estimate_pose_results[0], boxes):

            assert pose.shape == (17,pose_estimation_output_resolution[1],pose_estimation_output_resolution[0])
            
            # 0     : # nose
            # 1,2   : # eyes
            # 3,4   : # ears
            # 5,6   : # shoulders
            # 7,8   : # elbows
            # 9,10  : # wrists
            # 11,12 : # waist
            # 13,14 : # knees
            # 15,16 : # ankles

            joint_pos_in_camera_space = []

            for i_joint in range(17):
                x, y, score = self.find_highest( pose[i_joint] )
                #print( "joint[%d] : pos=(%d,%d), score=%8f" % (i_joint, x,y, score), flush=True )
                x_in_camera_image = ((box[0] + (box[2]-box[0]) * x / 48) * stream.image.shape[1] / people_detection_input_resolution[0]).item()
                y_in_camera_image = ((box[1] + (box[3]-box[1]) * y / 64) * stream.image.shape[0] / people_detection_input_resolution[1]).item()
                joint_pos_in_camera_space.append( np.array([ int(x_in_camera_image), int(y_in_camera_image) ]) )

            # shoulders
            cv2.line( stream.image, joint_pos_in_camera_space[5], joint_pos_in_camera_space[6], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # shoulders - elbows
            cv2.line( stream.image, joint_pos_in_camera_space[5], joint_pos_in_camera_space[7], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )
            cv2.line( stream.image, joint_pos_in_camera_space[6], joint_pos_in_camera_space[8], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # elbows - wrists
            cv2.line( stream.image, joint_pos_in_camera_space[7], joint_pos_in_camera_space[9], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )
            cv2.line( stream.image, joint_pos_in_camera_space[8], joint_pos_in_camera_space[10], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # waist
            cv2.line( stream.image, joint_pos_in_camera_space[11], joint_pos_in_camera_space[12], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # waist - knees
            cv2.line( stream.image, joint_pos_in_camera_space[11], joint_pos_in_camera_space[13], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )
            cv2.line( stream.image, joint_pos_in_camera_space[12], joint_pos_in_camera_space[14], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # knees - ankles
            cv2.line( stream.image, joint_pos_in_camera_space[13], joint_pos_in_camera_space[15], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )
            cv2.line( stream.image, joint_pos_in_camera_space[14], joint_pos_in_camera_space[16], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # center of sholders - waist
            center_of_sholders = (joint_pos_in_camera_space[5] + joint_pos_in_camera_space[6]) // 2
            center_of_waist = (joint_pos_in_camera_space[11] + joint_pos_in_camera_space[12]) // 2
            cv2.line( stream.image, center_of_sholders, center_of_waist, color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # center of sholders - nose
            cv2.line( stream.image, center_of_sholders, joint_pos_in_camera_space[0], color=line_color, thickness=line_thickness, lineType=cv2.LINE_8 )

            # dots on all joints
            for p in joint_pos_in_camera_space:
                cv2.rectangle( stream.image, p-dot_size, p+dot_size, color=dot_color, thickness=-1 )

    # Find a pixel in the heatmap (to find joint position)
    def find_highest( self, heatmap ):

        # FIXME : can I optimize this heavy process?

        assert heatmap.shape == (pose_estimation_output_resolution[1],pose_estimation_output_resolution[0])
        
        highest = None

        for y in range(pose_estimation_output_resolution[1]):
            for x in range(pose_estimation_output_resolution[0]):
                if highest==None:
                    highest = ( x, y, heatmap[y][x] )
                elif heatmap[y][x] > highest[2]:
                    highest = ( x, y, heatmap[y][x] )

        return ( highest[0], highest[1], heatmap[y][x].item() )

    # Trigger one-time profiling on the next frame
    def command_Profile(self):
        self.profile_requested = True

    # Trigger taking screenshot(s)
    def command_Screenshot( self, frames ):
        self.screenshot_requested_frames = frames
        self.screenshot_current_frame = 0


# Create application instance and run
def main():
    
    node = panoramasdk.node()

    app = Application(node)
    app.run()


if __name__ == "__main__":
    main()

