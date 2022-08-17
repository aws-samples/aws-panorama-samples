import boto3
import sagemaker
import os
import pandas as pd
import json
import ast
import cv2
import os
import numpy


class download_from_labeling_job(object):
    
    def __init__(self, init_object):
        #self.jobid = init_object.getval('job_name')
        self.jobid = init_object
    
    def download_s3_folder(self, bucket_name, s3_folder, local_dir=None):
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            if obj.key.split('/')[-1].split('.')[-1].lower() in ['png','jpg','jpeg'] :
                target = obj.key if local_dir is None \
                    else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                if obj.key[-1] == '/':
                    continue
                bucket.download_file(obj.key, target)

    def download_s3_data(self):

        client = boto3.client('sagemaker')
        s3 = boto3.client('s3')
        
        if not os.path.exists('./object_detection/prep_data/data'):
            os.makedirs('./object_detection/prep_data/data')
        
        # deleting everything in the folder first
        os.system("rm ./object_detection/prep_data/data/*")

        #describe the job id

        response = client.describe_labeling_job(
            LabelingJobName=self.jobid
        )

        # Download output.manifest

        manifest_file = response['LabelingJobOutput']['OutputDatasetS3Uri']
        list_manifest = manifest_file.split('/')
        bucket_name = list_manifest[2]

        filepath = ''
        s3_list = list_manifest[3:]
        for val in range(len(s3_list)):
            if val < len(s3_list) - 1:
                filepath += s3_list[val] + '/'
            else:
                filepath += s3_list[val]

        filename = './object_detection/prep_data/data/' + list_manifest[-1]
        s3.download_file(bucket_name, filepath, filename)

        # download data

        folder = response['InputConfig']['DataSource']['S3DataSource']['ManifestS3Uri'].split('dataset')[0]
        bucket_name_data = folder.split("/")[2]

        s3_folder = ''
        s3_list = folder.split("/")[3:]
        for val in range(len(s3_list)):
            if val < len(s3_list) - 1:
                s3_folder += s3_list[val] + '/'
            else:
                s3_folder += s3_list[val]

        self.download_s3_folder(bucket_name_data, s3_folder,'./object_detection/prep_data/data')

        return response
    
    
    def split_images_test_train_valid(self):
        GT_Job_Name = self.jobid
        if not os.path.exists('./{}'.format(GT_Job_Name)):
            os.makedirs('./{}'.format(GT_Job_Name))

        if not os.path.exists('./{}/test'.format(GT_Job_Name)):
            os.makedirs('./{}/test'.format(GT_Job_Name))

        if not os.path.exists('./{}/test/images'.format(GT_Job_Name)):
            os.makedirs('./{}/test/images'.format(GT_Job_Name))

        if not os.path.exists('./{}/test/labels'.format(GT_Job_Name)):
            os.makedirs('./{}/test/labels'.format(GT_Job_Name))

        if not os.path.exists('./{}/train'.format(GT_Job_Name)):
            os.makedirs('./{}/train'.format(GT_Job_Name))

        if not os.path.exists('./{}/train/images'.format(GT_Job_Name)):
            os.makedirs('./{}/train/images'.format(GT_Job_Name))

        if not os.path.exists('./{}/train/labels'.format(GT_Job_Name)):
            os.makedirs('./{}/train/labels'.format(GT_Job_Name))

        if not os.path.exists('./{}/valid'.format(GT_Job_Name)):
            os.makedirs('./{}/valid'.format(GT_Job_Name))

        if not os.path.exists('./{}/valid/images'.format(GT_Job_Name)):
            os.makedirs('./{}/valid/images'.format(GT_Job_Name))

        if not os.path.exists('./{}/valid/labels'.format(GT_Job_Name)):
            os.makedirs('./{}/valid/labels'.format(GT_Job_Name))
    
    
    def convert_to_yolov5_format(self, filename, file_annotations):
        GT_Job_Name = self.jobid
        result = []
        for annotation in file_annotations:
            annotations1 = annotation
            class_id = annotations1['class_id']
            x1 = annotations1['top']
            y1 = annotations1['left']
            width = annotations1['width']
            height = annotations1['height']
            x2 = x1+height
            y2 = y1+width
            coords = (y1, y2, x1, x2)
            x_center,y_center = ( numpy.average(coords[:2]),numpy.average(coords[2:]))
            image = cv2.imread('./object_detection/prep_data/data/{}'.format(filename))
            image = cv2.rectangle(image, (int(y1), int(x1)), (int(y2), int(x2)), (255,0,0), 2)
            image = cv2.circle(image, (int(x_center),int(y_center)), radius=0, color=(255,0,0), thickness=3)
            x_center = x_center/image.shape[1]
            y_center = y_center/image.shape[0]
            width = width/image.shape[1]
            height = height/image.shape[0]
            result.append([class_id, x_center, y_center, width, height])
        return result
    
    def create_train_test_valid(self):
        GT_Job_Name = self.jobid
        #manifest  = pd.read_csv('./object_detection/prep_data/data/output.manifest',sep=" ", header=None)
        with open('./object_detection/prep_data/data/output.manifest') as f:
            lines = f.readlines()
        manifest = pd.DataFrame(lines)
        train = int(len(manifest) * 80 / 100)
        test = int(train+1) + int(len(manifest) * 10 / 100)
        valid = int(test + 1) + int(len(manifest) * 10 / 100)

        for val in range(len(manifest)):
            row = ast.literal_eval(manifest.iloc[val][0])
            filename = row['source-ref'].split('/')[-1]
            annotations = row[GT_Job_Name]['annotations']
            result = self.convert_to_yolov5_format(filename, annotations)
            if val <= train:
                folder='train'
            elif val > train and val <= test:
                folder='test'
            else:
                folder='valid'

            # Move file and write file
            #move
            os.system('cp "./object_detection/prep_data/data/{}" "./{}/{}/images"'.format(filename,GT_Job_Name,folder))
            # write file
            with open('./{}/{}/labels/{}.txt'.format(GT_Job_Name,folder,filename.split('.')[0]), 'w') as f:
                for annot in result:
                    line = ''
                    line += str(annot[0])
                    line += ' '
                    line += str(annot[1])
                    line += ' '
                    line += str(annot[2])
                    line += ' '
                    line += str(annot[3])
                    line += ' '
                    line += str(annot[4])
                    f.write(line)
                    f.write('\n')
        
        os.system('rm -r ./object_detection')
    
    def get_class_map(self):
        GT_Job_Name = self.jobid
        #manifest  = pd.read_csv('./object_detection/prep_data/data/output.manifest',sep=" ", header=None)
        with open('./object_detection/prep_data/data/output.manifest') as f:
            lines = f.readlines()
        manifest = pd.DataFrame(lines)
        row = ast.literal_eval(manifest.iloc[0][0])
        output = [row[GT_Job_Name+'-metadata']['class-map'][str(x)] for x in row[GT_Job_Name+'-metadata']['class-map']]
        return output
        
    def start(self):
        return self.download_s3_data()
    
    

class communication(object):
    
    def __init__(self):
        pass
    
    def get_queue_url(self, name):
        sqs_client = boto3.client("sqs")
        response = sqs_client.get_queue_url(
            QueueName=name,
        )
        return response["QueueUrl"]
    
    def create_queue(self, name):
        sqs_client = boto3.client("sqs")
        response = sqs_client.create_queue(
            QueueName=name,
            Attributes={
                "DelaySeconds": "0",
            }
        )
        return response
        
    def send_message(self, qurl, message):
        sqs_client = boto3.client("sqs")

        response = sqs_client.send_message(
            QueueUrl=qurl,
            MessageBody=json.dumps(message)
        )
        # print(response)
        return response
    
    def receive_message(self, qurl):
        sqs_client = boto3.client("sqs")
        response = sqs_client.receive_message(
            QueueUrl=qurl,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=10,
        )

        print(f"Number of messages received: {len(response.get('Messages', []))}")
           
        messeges = []
        for message in response.get("Messages", []):
            message_body = message["Body"]
            print(f"Message body: {json.loads(message_body)}")
            #messeges.append({json.loads(message_body)})
            messeges.append([message["Body"], message["ReceiptHandle"]])
            #messeges.append({json.loads(message["ReceiptHandle"])})
        
        return messeges
    
    def empty_queue(self,qurl, receipt_handle):
        sqs_client = boto3.client("sqs")
        try:
            response = sqs_client.delete_message(QueueUrl=qurl,
                                             ReceiptHandle=receipt_handle)
        except ClientError:
            logger.exception(
                f'Could not delete the meessage from the - {qurl}.')
            raise
        else:
            return response


        
        
        






