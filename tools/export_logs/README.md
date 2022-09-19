## Overview

**panorama_export_logs.py** is a Python script to export Panorama related log streams, and archive them as a Zip file. You can download all the relevant logs locally to browse them using your preferred text editor, or you can share the Zip archive file with your team mates or AWS support team when you ask troubleshooting, even when you are not sure which log stream contains important information.


## Features

* By specifying device-id and application-instance-id, the tool automatically identifies log groups and log streams to download. (application-instance-id is optional if application level logs are not needed.)
* You can specify a date-time range (in UTC) to export, and export logs of multiple days quickly.
* It creates a Zip file (e.g. "panorama_exported_logs_20220720_145056.zip") at your current working directory. It contains all the log streams based on the device-id and application-instance-id you specified.
* The tool creates "info.json" file in the Zip file, which contains account id and region name, for easer issue reporting.


## How to use

1. Prepare a S3 bucket and prefix this tool uses as a working place.
2. Identify your device-id and application-instance-id to export logs.
3. Open a Terminal window, and use this tool.

    ```
    $ python3 panorama_export_logs.py --help
    usage: panorama_export_logs.py [-h] [--region REGION] [--device-id DEVICE_ID]
                                [--app-id APP_ID] [--s3-path S3_PATH]
                                [--start-datetime START_DATETIME]
                                [--end-datetime END_DATETIME]

    Export Panorama device level / application level logs in a Zip file

    optional arguments:
    -h, --help            show this help message and exit
    --region REGION       Region name such as us-east-1
    --device-id DEVICE_ID
                            Panorama device-id
    --app-id APP_ID       Panorama application instance id
    --s3-path S3_PATH     S3 path as a working place
    --start-datetime START_DATETIME
                            Start date-time in UTC, in YYYYMMDD_HHMMSS format
    --end-datetime END_DATETIME
                            End date-time in UTC, in YYYYMMDD_HHMMSS format
    ```


## Permission setting

To use this tool, you need to have permissions to 1) create and describe log export tasks, 2) list and get S3 objects. Following is an example of IAM policy. (**Note** : please replace the account-id and bucket name parts.)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "logs:DescribeExportTasks",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateExportTask",
                "s3:GetObject",
                "s3:GetObjectAttributes",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:logs:*:123456789012:log-group:/aws/panorama/*",
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

In addition to IAM permission, you need to configure the S3 bucket policy to allow CloudWatch Logs can export logs to the bucket. (**Note** : please replace the region-name, bucket name parts.)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "logs.us-east-1.amazonaws.com"
            },
            "Action": "s3:GetBucketAcl",
            "Resource": "arn:aws:s3:::your-bucket-name"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "logs.us-east-1.amazonaws.com"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::your-bucket-name/*",
            "Condition": {
                "StringEquals": {
                    "s3:x-amz-acl": "bucket-owner-full-control"
                }
            }
        }
    ]
}
```

For more details about permissions, please refer to [this document](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/S3ExportTasksConsole.html#S3PermissionsConsole).


## Sample usage

With device-id only (application level logs are not exported) :
```
$ python3 panorama_export_logs.py \
  --device-id device-u3amjjs5duop656pufpbz23abc \
  --s3-path s3://my-exported-logs/my-panorama-project \
  --start-datetime 20220621_000000 \
  --end-datetime 20220702_000000
```

With device-id and application-instance-id (both device level logs and application level logs are exported) :
```
$ python3 panorama_export_logs.py \
  --device-id device-u3amjjs5duop656pufpbz23abc \
  --app-id applicationInstance-adamjopanh4ug7ijaq2xrwvdef \
  --s3-path s3://my-exported-logs/my-panorama-project \
  --start-datetime 20220621_000000 \
  --end-datetime 20220702_000000
```


## Output structure

This tool creates a Zip file at your current working directory, and the contents of the Zip file is structured as follows:

- panorama_exported_logs_YYYYMMDD_HHMMSS.zip
    - applicationInstance-* : directory for application level logs
        - *.log : application level log files
    - device-* : directory for device level logs
        - *.log : device level log files
    - info.json : associated information for exported logs (account-id, region-name)


## Limitations

* Currently this tool doesn't automatically clean up the exported S3 objects in the specified S3 bucket / prefix. Please clean up manually as-needed.
