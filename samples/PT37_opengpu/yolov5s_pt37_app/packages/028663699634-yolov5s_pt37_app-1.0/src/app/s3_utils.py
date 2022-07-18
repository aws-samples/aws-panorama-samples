import boto3
s3 = boto3.resource('s3')

def send_to_s3(self, image_to_send, bucket_name, filename):
    """
    Send Image to S3 bucket of Choice. Must be on the same account / Or AWS Credentials Defined in the begnining.
    Args: 
        Image to Send.
        Bucket Name.
        Filename of Choice.
    Returns:
        Boolean.
    """
    data_serial = cv2.imencode('.png', image_to_send)[1].tostring()
    s3.Object(bucket_name, filename).put(Body=data_serial,ContentType='image/PNG')
    return True