import sys
import io
import time
import threading
import gc
import traceback
import http.server
import socketserver

# return numbers of python objects for each type in a string
def get_py_object_stat():

    objs = gc.get_objects()
    stat = {}

    for obj in objs:
        str_type = str(type(obj))
        if str_type not in stat:
            stat[str_type] = 0
        stat[str_type] += 1

    keys = list( stat.keys() )
    keys.sort()

    max_len = 10
    for k in keys:
        if max_len < len(k):
            max_len = len(k)
    
    buf = io.StringIO()

    for k in keys:
        buf.write( "  %s%s : %d\n" % ( k, ' '*(max_len-len(k)), stat[k] ) )
    
    return buf.getvalue()


# return call-stacks of all python threads in a string
def get_py_threads():
    
    buf = io.StringIO()

    for th in threading.enumerate():
        buf.write( th.name + ":\n\n" )
        traceback.print_stack( sys._current_frames()[th.ident], limit=30, file=buf )
        buf.write( "\n---\n" )

    return buf.getvalue()


# Custom HTTP request handler for some introspection features
class IntrospectionHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):

        print( "GET :", self.path, flush=True )
        
        if self.path=="/py_object_stat":

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()

            message = get_py_object_stat()
            message = message.encode("utf-8")

            self.wfile.write(message)
            
        elif self.path=="/py_threads":

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()

            message = get_py_threads()
            message = message.encode("utf-8")

            self.wfile.write(message)
            
        else:
            self.send_response(404, "Resource doesn't exist")
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            
            message = "Resource Not found"
            message = message.encode("utf-8")

            self.wfile.write(message)

# HTTP server thread to serve for introspection requests
class IntrospectionHttpServerThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__( self, name="IntrospectionHttpServerThread" )
        self.canceled = False

    def run(self):
    
        print( "IntrospectionHttpServerThread started", flush=True )

        PORT = 8080

        with socketserver.TCPServer(("", PORT), IntrospectionHttpRequestHandler) as httpd:

            print( f"Starting HTTP request handler at port={PORT}", flush=True )

            httpd.timeout = 1
            while not self.canceled:
                httpd.handle_request()

    def cancel(self):
        self.canceled = True

# application class
#class Application(panoramasdk.node):
class Application:
    
    # initialize application
    def __init__(self):
        
        super().__init__()
        
        self.frame_count = 0

        # Start a http server thread
        self.http_server_thread = IntrospectionHttpServerThread()
        self.http_server_thread.start()

    # run top-level loop of application  
    def run(self):
        
        while True:
            
            # get video frames from camera inputs 
            #media_list = self.inputs.video_in.get()
            
            print("Frame :", self.frame_count, flush=True )

            # put video output to HDMI
            #self.outputs.video_out.put(media_list)
            
            self.frame_count += 1
            
            time.sleep(1)


app = Application()

try:
    app.run()
except KeyboardInterrupt:
    app.http_server_thread.cancel()
    app.http_server_thread.join()

