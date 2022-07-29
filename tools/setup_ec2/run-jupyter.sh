#!/bin/bash -xe

# without port forwarding, you need to allow which IP address you accept. 0.0.0.0 means any IP address.
#jupyter-lab --no-browser --allow-root --ip 0.0.0.0 --port 8888 --notebook-dir ~

# with port forwarding, you don't have to allow any IP addresses
jupyter-lab --no-browser --allow-root --port 8888 --notebook-dir ~
