# Panorama MJPEG Server

## License

Copyright (c) 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
This source code is subject to the terms found in the AWS Enterprise Customer Agreement.

## What is this?
The MJPEG server allows a user on the local Panorama network to view the input stream using a browser. The input stream can be manipulated using OpenCV to put bounding boxes and labels to provide an exact copy of the HDMI display to a local user via the browser.

## Usage

Import the MJPEG server and create an instance in the init routine:
```python
from mjpeg_server import PanoramaMJPEGServer

def init(self, parameters, inputs, outputs):
  # At the end of init
  self.mjpegserver = PanoramaMJPEGServer(host='0.0.0.0', port=9000)
```

Manipulate stream.image with bounding boxes and labels. At the end, feed the frame into the mjpeg server
```python
def entry(self, inputs, outputs):

  # Feed into mjpeg server after manipulating image
  self.mjpegserver.feed_frame(stream.image)

```

Add the following line to the ```enable_ssh()``` function in ```/etc/init.d/mht_boot_init```:

```bash
iptables -I INPUT -p tcp --dport 9000 -j ACCEPT
```

## Parameters to adjust (mjpeg_server.py)
Update the mjpg file:
```python
mjpeg_file='preview.mjpg'
```

Update how much to wait between each serving frame
```python
sleep_rate = 0.05
```

## How to view
Once done, point your favorite browser to:
http://panoramaip:9000/preview.mjpg