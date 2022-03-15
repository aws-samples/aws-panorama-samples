import panoramasdk

# application class
class Application(panoramasdk.node):
    
    # initialize application
    def __init__(self):
        
        super().__init__()
        
        self.frame_count = 0

        # TODO : implement here            

    # run top-level loop of application  
    def run(self):
        
        while True:
            
            # get video frames from camera inputs 
            media_list = self.inputs.video_in.get()
            
            # TODO : implement here            

            # put video output to HDMI
            self.outputs.video_out.put(media_list)
            
            self.frame_count += 1

app = Application()
app.run()
