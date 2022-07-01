'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import argparse
import textwrap
import sys
from os.path import exists
import time
import json
import boto3

from workflow.application_deployment_workflow import ApplicationDeploymentWorkflow
from workflow.utils import Logger
from workflow.utils import DeviceValidator

logger = Logger().get_logger()

termination_arg_table = {
    'Register': 1,
    'Deploy': 2,
    'Remove': 3
}

output_template = { "ValidationResult": [] }

if __name__ == '__main__':
    # read input argument
    parser = argparse.ArgumentParser(
        prog='panorama-csv-tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            This is the assistance tool for panorama camera stream validation.
            In order to validate. your data sources this script will create (register) data sources if they do not yet exist.
            An application will then be deployed on the device to confirm that the data sources (cameras) connection can be established. 
            Once deployment has completed and results for each data source have been returned the application will be removed from your device.
            It is important to note that the data source creation, application deployment and removal are operations that may take minutes to complete.
            '''),
        epilog=textwrap.dedent('''\
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
                ''')
    )
    # input file
    parser.add_argument('-i', '--input',
                        help='JSON file that contains the list of data source inputs.',
                        type=str,
                        )
    # configure flow
    # create camera package | deploy application | remove application
    # by default the whole flow is followed
    parser.add_argument('-t', '--termination',
                        choices=['Register', 'Deploy', 'Remove'],
                        default='Remove',
                        help='''
                        Stop the validation workflow at the step specified,
                         1) Register: stop flow after camera package registered
                         2) Deploy: stop flow after validation application deployed
                         3) Remove: stop flow after validation application removed (this is the default behavior) 
                        ''')
    # query non deleted validation
    parser.add_argument('-l', '--list',
                        action='store_true',
                        help='''\
                             Returns a list of all data sources and their validation results. 
                             This can only be used when you decide to terminate the script using the \"-t Deploy\" command.''')
    # parallel processing
    parser.add_argument('-n', '--number',
                        type=int,
                        default=8,
                        help='''\
                            Specify the number of cameras to be grouped in one validation application deployment. 
                            N must be a value of 8 or less. 
                            By default the panorama media service limit is 8 data sources (default) per application deployment. 
                            This means that if your JSON has more than 8 data sources to validate we will queue multiple 
                            application deployments (this will greatly increase the time needed to complete the full validation operation).
                            ''')

    # output path
    parser.add_argument('-o', '--output',
                        type=str,
                        help='''\
                            Export a JSON file which will contain the validation results. Specify the name only, e.g. "-o csvResults" will export a file named "csvResults.json".
                            ''')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # ========== Start processing args =============
    if args.input is None:
        logger.error("Please specify the input file! Check the sample input from --help")
        sys.exit(-1)

    output_file = ""
    if args.output is not None:
        output_file = args.output
        if not args.output.endswith("json"):
            output_file = output_file + ".json"

        if not exists(output_file):
            logger.info("Output path: {} does not exist, creating now".format(output_file))
            with open(output_file, "w+") as fd:
                json.dump(output_template, fd, indent=4)
        else:
            logger.info("Output path: {} specified.".format(output_file))

    input_file = json.load(open(args.input))

    account_id = boto3.client('sts').get_caller_identity().get('Account')
    if 'device_id' not in input_file:
        device_id = DeviceValidator().select_device()[1]
    else:
        device_id = input_file['device_id']

    data_sources = []
    for data_source in input_file['cameras']:
        version = "1.0" if "version" not in data_source else data_source['version']
        remove = False if "remove" not in data_source else data_source['remove']
        print(data_source.get('Username'))
        data_sources.append( (data_source['name'],
                              data_source.get('Username'),
                              data_source.get('Password'),
                              data_source.get('StreamUrl'),
                              version,
                              remove) )

    if len(data_sources) == 0:
        logger.warning("No data source to be validated, exit")
        sys.exit(0)

    logger.info("Prepare camera stream validation against device {} from account {}".format(device_id, account_id))
    workflow = ApplicationDeploymentWorkflow(account_id, device_id, output_file)

    if args.list is True:
        logger.info("List application validation result")
        workflow.list_deployed_application()
        sys.exit(0)

    total_deployment = (len(data_sources) + args.number - 1) // args.number
    logger.info("There are total {} cameras to be validated with {} data sources to be grouped in one deployment, {} deployments will be run sequentially.".format(
        len(data_sources), args.number, total_deployment)
    )

    start_time = time.time()
    # register data source first to avoid json error during latter deployment
    workflow.data_source_check_and_create(data_sources)

    cur = 0
    for i in range(total_deployment):
        logger.info("======= Deployment {} =======".format(i + 1))
        cur_data_sources = data_sources[cur: cur + args.number]
        workflow.run(termination_arg_table[args.termination], cur_data_sources)
        cur += args.number

    workflow.data_source_removal(data_sources)
    end_time = time.time()
    logger.info("It takes {} secs for camera stream validation flow finished!".format(end_time - start_time))

