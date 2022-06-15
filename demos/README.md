# Demo applications for Panorama

In this directory, we maintain multiple demo applications. The purpose of demo applications is to showcase how Panorama can solve busness problems, rather than explaining how to develop applications or how to use technical features.

For more details of demo applications, please see README files in specific demo application directories.

In this document, we also write how to set up and run these demo applications on real Panorama devices including required equipments and deployment instructions.


### **PanoJupyter as a foundation of demo applications**

Demo applications use [PanoJupyter](https://github.com/aws-samples/aws-panorama-samples/tree/main/tools/pano_jupyter) as the common foundation. Each demo application is implemented as a notebook which runs on PanoJupyter.



### **Required equipments**

To set up demo application in conference events, in-person customer meeting, etc, please make sure following equipments are prepared either by you or your customer.

* Panorama appliance device x 1
    * **Important** : Make sure the device is configured with DHCP (not static IP setting).
    * **Important** : Make sure the device has latest firmware version, in order to avoid taking time to firmware update in the demo place.
    * **Recommended** : Consider preparing another Panorama appliance device for back up
* IP Camera(s) which support RTSP with H.264.
    * [Supported cameras](https://docs.aws.amazon.com/panorama/latest/dev/gettingstarted-compatibility.html#gettingstarted-compatibility-cameras)
    * **Important** : Make sure the cameras are configured with DHCP (not static IP setting).
    * **Recommended** : Consider preparing additional IP cameras for any hardware troubles, compatibility issue, etc.
* LAN cables x (3 + number of cameras)
    * **Note** : LAN cables are needed for 1) Panorama device, 2) Cameras 3) Laptop, 4) Uplink.
    * **Important** : Make sure length of LAN cables is enough, especially for cameras.
* HDMI display x 1
* HDMI cable x 1
* Network switch with PoE feature x 1
    * **Note** : Number of ports has to cover at least 1) number of Panorama devices, 2) number of cameras, 3) Laptop, and 4) Up-link to internet.
    * **Important** : Many IP cameras support PoE (Power over Ethernet). Make sure the Network switch has enough PoE ports for cameras.
* FAT32 formatted USB memory x 1
    * Just in case you need to re-provision the device.
* Laptop x 1
    * To create data sources on Management Console, deploy application to the device, and operate PanoJupyter over web browser.
* USB-C hub x 1
    * If your Laptop doesn't have Ethernet port or USB-A port and only has USB-C port, prepare USB-C hub.
* Power strips x 1


### **Network requirement**

In order to set up demo in different network (conference event, customer's office, etc), you need to re-deploy the PanoJupyter application, because IP addresses of cameras change and you cannot change IP addresses of cameras without re-deploying application.

Also in order to deploy the PanoJupyter application, the device needs to have access to AWS cloud services.

* Access to internet
    * Make sure your Laptop and Panorama device can access internet.
    * You can confirm this by visiting Amazon.com website on web browser.
* Access to AWS services
    * Make sure your Laptop and Panorama device can access AWS services.
    * You can confirm this by hitting "aws s3 ls" command in the terminal.
* Download bandwidth higher than 20 Mbps
    * If you don't have enough network bandwidth, you need to spend longer time to deploy PanoJupyter application.


### Set up instructions

1. Connect Network switch, Cameras, and Laptop to the network.
1. Check network connectivity to AWS cloud.
    * You can confirm this by visiting Amazon.com website on browser, and hitting "aws s3 ls" command in terminal.
1. Check network bandwidth.
    * By googling "network speed test", you can find websites you can test network bandwidth.
1. Find Camera IP addresses.
    To discover IP addresses of cameras, you can use following application.
    * Windows : [ONVIF Device Manager](https://sourceforge.net/projects/onvifdm/)
    * MacOS : TBD
1. Confirm RTSP streams from cameras are visible on Laptop
    * With VLC media player.
1. Create data source(s) on Panorama Management Console UI
    * [Console Link](https://console.aws.amazon.com/panorama/home#data-sources)
1. Connect HDMI display with Panorama device, plug LAN cables in Panorama device, and connect power cable.
    * Panorama device boots up automatically when power cable is pluged in.
1. Confirm that Panorama splash screen and boot-up sequence logs are displayed on the display.
    * **Important** : Make sure HDMI is connected as of powering the device on. There is a bug where device doesn't try to use HDMI if it is connected after powering on.
1. Deploy PanoJupyter app and open it.
    * Follow PanoJupyter deployment instruction in [this page](https://github.com/aws-samples/aws-panorama-samples/tree/main/tools/pano_jupyter).
    * **Important** : Make sure you opt-in inbound networking, and configure port number.
1. Create a directory for a demo (e.g. “heatmap”. you can choose any name) on PanoJupyter, Drag & Drop notebook file (and other required files if any) in the directory.
1. Open the notebook, and run notebook by choosing “Run > Run All Cells” from the menu bar.
1. Monitor the notebook and HDMI display, confirm the demo runs successfully.
    * Depending on the demo, it may take some time to load model data, and run first inference call. (30 sec ~ 10 min)
1. Shutdown and reboot the device by pressing power button before running another demo.
