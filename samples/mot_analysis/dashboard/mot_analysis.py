FPS = 5
BUCKET_NAME = '##BUCKET_NAME##'
KVS_STREAM_NAME = "panorama_preview"
KVS_TIMEOUT = 3600  # 1 hour of web session time

import awswrangler as wr  
import boto3
import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
from streamlit_autorefresh import st_autorefresh

from streamlit_img_label import st_img_label, annotation
import PIL
from PIL import ImageDraw
from heatmap import Heatmapper

#from streamlit_player import st_player
import streamlit.components.v1 as components

st.set_page_config(page_title=None, 
                    page_icon=None, 
                    layout="wide", 
                    initial_sidebar_state="auto", 
                    menu_items=None)

matplotlib.rc('legend', fontsize=20) # using a size in points
matplotlib.rc('xtick', labelsize=5) 
matplotlib.rc('ytick', labelsize=5)
matplotlib.rc('axes', titlesize=15, labelsize=15)   #For setting chart title
matplotlib.rc('figure', titlesize=20)
matplotlib.rc('font', size=15)

def CheckTime(): return datetime.datetime.now()

session = boto3.Session()

s3 = session.client('s3')
response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix ='dailycapture/', MaxKeys=100)
cameras = []
for object in response['Contents']:
    name = object['Key'].split('/')[1]
    if name != '':
        cameras.append(name)

cameras = list(set(cameras))

@st.experimental_singleton
def get_data(query):
    return wr.athena.read_sql_query(sql=query, database="default", boto3_session=session)

#Fix me: Fix partition when first loading
#if st_autorefresh(interval=60000, key="refresh") == 0:
#    wr.athena.read_sql_query(sql=f"MSCK REPAIR TABLE `heatmap`;", database="default", boto3_session=session)

camera_id = st.sidebar.selectbox('Select camera', cameras)
st.sidebar.text("\n")
today = datetime.datetime.now(datetime.timezone.utc)
dt = st.sidebar.date_input("Pick a day to render", today)
st.sidebar.text("\n")
start_time, end_time = map(int, st.sidebar.select_slider('Select a range of time in hour',
     options=[v for v in range(0,25)],
     value=(0, 24)))
st.sidebar.text("\n")
query = f"SELECT distinct \"cid\" FROM heatmap where \"camera\"='{camera_id}' and year='{dt.year}' and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}' order by \"cid\""
cid = list(get_data(query)['cid'])
cococategory = np.array(['Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat'])
tclist = st.sidebar.multiselect('Tracking category', list(cococategory[cid]), list(cococategory[cid]) if len(cid) > 0 else None)
if len(tclist) > 0:
    cid = [i for i,l in enumerate(cococategory) if l in tclist]
    query = f"SELECT \"tid\", count(\"tid\") as \"fcount\" FROM heatmap where \"camera\"='{camera_id}' and year='{dt.year}' and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}' and \"cid\" in {tuple(cid)} group by \"tid\" order by \"fcount\" desc limit 10"
    tclist = tuple(list(get_data(query)['tid']))
else:
    tclist = []
st.sidebar.text("\n")
tidlist = st.sidebar.multiselect('Tracking history', tclist)
if len(tidlist) > 0:
    query = f"SELECT \"tid\", \"ts\", \"fnum\", (\"left\" + \"w\"/2) as x, (\"top\" + \"h\") as y FROM heatmap where \"camera\"='{camera_id}' and tid in {tuple(tidlist + [0])} and year='{dt.year}' and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}' and \"cid\" in {tuple(cid)} order by \"fnum\""
    tid_df = get_data(query)
st.write(f'Rendering {camera_id}/{dt.year}-{dt.month}-{dt.day}, hour {start_time} to {end_time}')

import io
@st.experimental_singleton
def get_cameraimage(bucket, key):
    file_byte_string = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    return PIL.Image.open(io.BytesIO(file_byte_string))

import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.transform

@st.experimental_singleton
def get_renderedheatmap(_camera, df, alpha=0.5, cmap='viridis', axis='off'):
    if len(df) == 0:
        return None
    x = np.zeros((2001, 2001)) #heatmap base
    for row in df.iterrows():
        x[int(row[1].y*2000), int(row[1].x*2000)] = 1 #center of heat
    heat_map = ndimage.filters.gaussian_filter(x, sigma=32)

    # resize heat map
    heat_map_resized = skimage.transform.resize(heat_map, (_camera.height, _camera.width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    figure = plt.gcf()
    figure.set_size_inches(16, 10)
    
    # display
    plt.imshow(_camera)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    buf.seek(0)
    return PIL.Image.open(buf)

try:
    try:
        kvs = session.client("kinesisvideo")
        # Grab the endpoint from GetDataEndpoint
        endpoint = kvs.get_data_endpoint(APIName="GET_HLS_STREAMING_SESSION_URL", StreamName=KVS_STREAM_NAME)['DataEndpoint']
        # Grab the HLS Stream URL from the endpoint
        kvam = session.client("kinesis-video-archived-media", endpoint_url=endpoint)
        url = kvam.get_hls_streaming_session_url(StreamName=KVS_STREAM_NAME, PlaybackMode="LIVE", Expires=KVS_TIMEOUT)['HLSStreamingSessionURL']
        
        embedded_player = f"<script src='https://cdn.jsdelivr.net/npm/hls.js@latest'></script> \
            <center><video id='video' class='player' width='100%' controls autoplay muted></video></center> \
            <script> \
                var video = document.getElementById('video'); \
                var hls = new Hls(); \
                hls.loadSource('{url}'); \
                hls.attachMedia(video); \
                hls.on(Hls.Events.MANIFEST_PARSED,function() {{ \
                    video.play(); \
                }}); \
            </script>"
        st.write(f'Live analysis')
        components.html(embedded_player, height=600)
    except:
        st.write("Video offline")
    
    #For getting heatmap
    image_name = f'{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}.png'
    camera = get_cameraimage(BUCKET_NAME, f'dailycapture/{camera_id}/{image_name}')
    #query = f"SELECT count, cast(hour as int) + 9 as hour, (\"top\" + \"h\") as x, (\"left\"+\"w\")/2 as y FROM heatmap where sid='{camera_id}' and year='{dt.year}' and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time+TIME_LOCALE).zfill(2)}' and '{str(end_time+TIME_LOCALE).zfill(2)}'"
    query = f"SELECT (\"left\" + \"w\"/2) as x, (\"top\" + \"h\") as y FROM heatmap where camera='{camera_id}' and year='{dt.year}' and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}' and \"cid\" in {tuple(cid)}"
    pos_df = get_data(query)
    pos_df = pos_df[(pos_df['x'] <= 1) & (pos_df['x'] >= 0) & (pos_df['y'] <= 1) & (pos_df['y'] >= 0)]
    heatImage = get_renderedheatmap(camera, pos_df).copy()

    if len(tidlist) > 0:
        draw = ImageDraw.Draw(heatImage)
        for id, tid in tid_df.groupby('tid'):
            draw.text((tid['x'].iloc[0] * heatImage.width, tid['y'].iloc[0] * heatImage.height),"ID: " + str(id),(255,255,255))
            draw.line(list(zip(tid['x'] * heatImage.width, tid['y'] * heatImage.height)), fill="red", width=3)
            draw.text((tid['x'].iloc[-1] * heatImage.width, tid['y'].iloc[-1] * heatImage.height),str((tid.iloc[-1]['fnum'] - tid.iloc[0]['fnum'])/FPS)+" sec",(255,255,255))

    rects = annotation.read_xml(f'{image_name}')
    st.write(f'Heatmap and tracking in given time range')
    rects = st_img_label(heatImage, box_color="yellow", rects=rects)

    #Optional rendering for cropped rects
    crop_heatmap, numpeople_chart, longstay_chart = st.columns([1, 2, 2])
    for idx, val in enumerate(rects):
        rects[idx]['label'] = crop_heatmap.text_input('Label', val['label'], key=idx)
        crop_heatmap.image(heatImage.crop((val['left'], val['top'], val['left'] + val['width'], val['top'] + val['height'])))
        left = val['left'] / heatImage.width
        right = (val['left'] + val['width']) / heatImage.width
        bottom = (val['top'] + val['height']) / heatImage.height
        top = val['top'] / heatImage.height
        query = f"SELECT cast(hour as int) as hour, count(distinct tid) as num FROM heatmap where camera='{camera_id}' and year='{dt.year}'" \
                f" and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}'" \
                f" and (\"left\" + \"w\"/2) between {left} and {right} and (\"top\" + \"h\") between {top} and {bottom} group by hour"
        #we should reflect group by frame number to uniquely identify number in that scene
        numpeople_df = get_data(query).set_index('hour')
        numpeople_chart.line_chart(numpeople_df)
        query = f"SELECT cast(hour as int) as hour, max(stay) as maxstay FROM " \
                f"(SELECT hour, ((array_agg(age order by tid, fnum desc))[1] - (array_agg(age order by tid, fnum asc))[1]) as stay from heatmap"\
                f" where camera='{camera_id}' and year='{dt.year}'" \
                f" and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}'" \
                f" and (\"left\" + \"w\"/2) between {left} and {right} and (\"top\" + \"h\") between {top} and {bottom} group by hour, tid) group by hour order by hour"
        longstay_df = get_data(query).set_index('hour')
        longstay_chart.line_chart(longstay_df)

    if st.button('Save analysis') == True:
        for idx, val in enumerate(rects):
            rects[idx]['left'] = round(val['left'])
            rects[idx]['top'] = round(val['top'])
            rects[idx]['width'] = round(val['width'])
            rects[idx]['height'] = round(val['height'])
        annotation.output_xml(f'{image_name}', heatImage, rects)
    
    #For getting maximum object in a single shot
    query = f"SELECT cast(hour as int) as hour, max(ocount) as Crowd FROM (SELECT hour, count(fnum) as ocount FROM heatmap where camera='{camera_id}' and year='{dt.year}' and month='{str(dt.month).zfill(2)}' and day='{str(dt.day).zfill(2)}' and hour between '{str(start_time).zfill(2)}' and '{str(end_time).zfill(2)}' and \"cid\" in {tuple(cid)} group by hour, fnum) group by hour order by hour"
    crowd_df = get_data(query).set_index('hour').reindex(list(range(start_time, end_time+1, 1)), fill_value=0)
    st.write(f'Busy hours')
    st.line_chart(crowd_df)

except Exception as e:
    st.write("No collected data")