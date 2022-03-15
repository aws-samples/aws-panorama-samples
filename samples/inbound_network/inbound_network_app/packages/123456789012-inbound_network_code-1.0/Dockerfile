# This Dockerfile fetches the image tagged latest by default
# To use a specific version of the image, refer to https://gallery.ecr.aws/panorama/panorama-application
# and tag the image in the Dockerfile with the version you're planning to use.
FROM public.ecr.aws/panorama/panorama-application
COPY src /panorama
RUN pip3 install boto3
