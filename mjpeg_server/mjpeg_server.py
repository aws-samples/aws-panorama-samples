# Copyright (c) 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# This source code is subject to the terms found in the AWS Enterprise Customer Agreement.
"""
MJPEG server for Panorama inference output
"""
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
import cv2
from threading import Thread
from threading import Lock
import time

#logging.getLogger().setLevel('DEBUG')

#Globals
display_buffer = None
frames_received = False
mjpeg_lock = Lock()
mjpeg_file = 'preview.mjpg'
sleep_rate = 0.05
mjpeg_path = '/' + mjpeg_file
print('MJPEG Path{}'.format(mjpeg_path))


class PanoramaMJPEGServerHandler(BaseHTTPRequestHandler):
  """
  Take frames from panorama inference output and serve it up as a rudimentery
  screen scraper in the form of an MJPEG server. NOTE: You have to add the labels
  and rectangles using OpenCV instead of the Panorama SDK for this to work
  """

  def do_GET(self):
    """
    Return mjpeg frames
    """

    global display_buffer, frames_received

    logging.debug("do_GET: {}", self.path)
    if self.path == mjpeg_path:
      try:

        # Return if frames have not been received
        if frames_received == False:
          self.send_file_not_found()
          return

        else:

          # Send 200 with the jpeg boundary
          self.send_response(200)
          self.send_header(
              'Content-type',
              'multipart/x-mixed-replace; boundary=--jpgboundary'
          )
          self.end_headers()

          # Sit in a forever loop and keep serving up frames
          while True:

            # Acquire lock
            logging.debug("Acquiring lock for jpg")
            mjpeg_lock.acquire()

            # Send the converted jpeg buffer
            self.wfile.write("--jpgboundary".encode("utf-8"))
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-length', str(len(display_buffer)))
            self.end_headers()
            self.wfile.write(display_buffer)

            # Release lock
            logging.debug("Releasing lock for jpg")
            mjpeg_lock.release()

            time.sleep(sleep_rate)

      except Exception as ex:
        logging.error("Error in mjpeg serve: %s", str(ex))
        mjpeg_lock.release()

    else:
      self.send_file_not_found()

  def send_file_not_found(self):
    """
    Send out 404 response
    """
    logging.debug("Sending File not Found")
    self.send_response(404)
    self.send_header('Content-type', 'text/html'.encode("utf-8"))
    self.end_headers()
    self.wfile.write('<html><head></head><body>'.encode("utf-8"))
    self.wfile.write('<h1>Frames not received</h1>'.encode("utf-8"))
    self.wfile.write('</body></html>'.encode("utf-8"))


class PanoramaMJPEGServer():
  """
  Panorama MJPEG server interface. Create instance in init()
  """
  def __init__(self,  host='0.0.0.0', port=9000):
    """
    Initialize HTTP server on port 9000. Note that you have to use the following command
    over SSH to serve up the frames
    iptables -I INPUT -p tcp --dport 9000 -j ACCEPT
    """
    self.host = host
    self.port = port

    # Start the http server in a thread
    self.server = HTTPServer((self.host, self.port),
                             PanoramaMJPEGServerHandler)
    self.server.allow_reuse_address = True
    self.http_thread = Thread(target=self.http_server_thread_function)
    self.http_thread.setDaemon(True)
    self.http_thread.start()

  def http_server_thread_function(self):
    """
    Run the http server in this thread
    """
    global frames_received

    self.server_started = True
    try:
      logging.info(
          'Server initialized at http://{}:{}{}'.format(self.host, self.port, mjpeg_path))
      self.server.serve_forever()
    except Exception as ex:
      logging.error("Error in httpserver: %s", str(ex))
      self.server_started = False
    finally:
      self.server.server_close()

  def feed_frame(self, display_array):
    """ Feed frame into the mjpeg server class """
    global display_buffer, frames_received

    try:
      # Don't serve until the first frame is received from panorama
      frames_received = True

      logging.debug("Acquiring lock for jpg")
      mjpeg_lock.acquire()
      logging.debug("Lock acquired")

      ret, jpegfile = cv2.imencode('.jpg', display_array)
      logging.debug("Return value when feeding frame: %s", ret)
      display_buffer = jpegfile.tostring()

    except Exception as ex:
        logging.error("Error in mjpeg feed frame: %s", str(ex))

    finally:
      mjpeg_lock.release()
      logging.debug("Lock released")
