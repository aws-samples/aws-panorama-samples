## Overview
**Camera Stream Validation(CSV)** is a Panorama tool for testing IP camera connection to specific Panorama device. Using the tool, you can anticipate the validation completes for about 5 minutes to get the status of each camera.

## Features
* **Register** the cameras if they are not registered in the AWS Panorama console
* Validate list of cameras through an application **Deployment** under the hood on the specific Panorama device
* Control the validation **stage** as the whole application deployment flow is executed for validation, and it can control in which stage e.g. register camera, deploy app or remove app, to terminate.
* List camera connection status at the moment when the associated application **remains** on device

## Security notice
This is the beta version software, please read the following agreement before using.

### AWS Beta Software Agreement
1. Customer agrees that any beta software provided by AWS Panorama will not be installed on any devices used in production settings or with production data.
2. Customer agrees that any beta software provided by AWS Panorama is provided as-is without any guarantees of reliability, performance or support.
3. Customer agrees that any beta software provided by AWS Panorama could stop working at any time. Customer is responsible for migrating workloads built on the beta software to the production software when the production software is available.
4. Customer agrees that any information regarding the beta feature is AWS Confidential information and access to the beta feature is limited to personnel who are pre-approved by AWS.

## Setup
Download the tool to your local PC/laptop, and then to configure the local environment to point to target accounts where your devices are provisioned.

```
1. aws configure
   // To make sure you have access to panorama service 
   AWS Access Key ID [********************]: Enter Access key
   AWS Secret Access Key [****************]: Enter Secret key
   Default region name [us-east-1]: Enter Region name
   Default output format [None]:  json or leave blank

2. specify the device_id[optional] information in the input json file
   device_id is optional, if not specified, you will be asked to select available devices listed from response.
```

## How to use
python3 workflow name:

Eg:

``` python3 main.py```  

Usage:

* Check usage by `python3 main.py -h`
* List current camera status from applications on device `python3 main.py -l`
* Normal validation `python3 main.py -i <input_json> -t Deploy`

```
usage: panorama-csv-tool [-h] [-i INPUT] [-t {Register,Deploy,Remove}] [-l] [-n NUMBER]

This is the assistance tool for AWS Panorama camera stream validation.
In order to validate. this script will create (register) data sources if they do not yet exist. An application will then be deployed on the device to confirm that the data sources (cameras) connection can be established. Once deployment has completed and results for each data source have been returned the application will be removed from your device.
It is important to note that the data source creation, application deployment and removal are operations that may take minutes to complete.

Optional arguments:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        JSON file that contains the list of data source inputs.
  -t {Register,Deploy,Remove}, --termination {Register,Deploy,Remove}
                        Stop the validation workflow at the step specified.
                        1) Register: stop the flow after all the data sources are created.
                        2) Deploy: stop the flow after the validation application is deployed to your device.
                        3) Remove: stop the flow after the validation application removed from your device (this is the default behavior)
  -l, --list            Returns a list of all data sources and their validation results. This can only be used when you decide to terminate the script using the "-t Deploy" command.
  -n NUMBER, --number NUMBER
                        Specify the number of cameras to be grouped in one validation application deployment. N must be a value of 8 or less. By default the panorama media service limit is 8 data sources (default) per application deployment. This means that if your JSON has more than 8 data sources to validate we will queue multiple application deployments (this will greatly increase the time needed to complete the full validation operation).

------------------------------------------------------------------------
Sample input
{
    // device_id is optional, if not specified, you will be asked to select available devices
    "device_id": "device-h3ai3ijd4w3nyi5tz5lbk5fvri" 
    "cameras": [
        {
            "name": "my-data-source",
            "Username": "admin",
            "Password": "admin",
            "StreamUrl": "rtsp://192.168.0.123:554/stream",
            "version": 1.0,
            // Determine remove this camera package after the flow
            "remove": false
        }
    ]
}

Sample output
{
    "ValidationResult": [
        {
            // auto-generated application id asociated with set of cameras validated
            "applicationInstance-ok75ymoq5tji2goxltqjqkpsuy": {
                "david_test_camera_4": "ERROR",
                "david_test_camera_1": "RUNNING"
            }
        }
    ]
}

```

## Limitations
* Camera registration and application may encounter timeout return depends on Panorama console response time and network status. 
* Camera validation status cannot be retrieved from removed application
* There is a camera state transition time between application gets deployed with success and camera is completed initialized, it may return camera is in error state if the camera is not setup property.