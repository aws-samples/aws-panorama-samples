### Setup
Configure the local environment to point to the expected region

```
1. aws configure
   // To make sure you have access to panorama service 
   AWS Access Key ID [********************]: Enter Access key
   AWS Secret Access Key [****************]: Enter Secret key
   Default region name [us-east-1]: Enter Region name
   Default output format [None]:  Leave blank

2. specify the device_id[optional] information in the input json file
   device_id is optional, if not specified, you will be asked to select available device listed from response.
```

### Execution
python3 workflowname:

Eg:

``` python3 main.py```  

Usage:

Check usage by `python main.py -h`

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
            "applicationInstance-ok75ymoq5tji2goxltqjqkpsuy": {
                "david_test_camera_4": "ERROR",
                "david_test_camera_1": "RUNNING"
            }
        },
        {
            "applicationInstance-3c76x7c47ua43irldj4hukbpx4": {
                "david_test_camera_4": "ERROR",
                "david_test_camera_1": "RUNNING"
            }
        }
    ]
}

```
### AWS Beta Software Agreement
1. Customer agrees that any beta software provided by AWS Panorama will not be installed on any devices used in production settings or with production data.
2. Customer agrees that any beta software provided by AWS Panorama is provided as-is without any guarantees of reliability, performance or support.
3. Customer agrees that any beta software provided by AWS Panorama could stop working at any time. Customer is responsible for migrating workloads built on the beta software to the production software when the production software is available.
4. Customer agrees that any information regarding the beta feature is AWS Confidential information and access to the beta feature is limited to personnel who are pre-approved by AWS.