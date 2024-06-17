# from flask import Flask
import random
import pika
import os
import cv2
import numpy as np
#import tensorflow as tf
import sys
import time
import glob
import socket
import datetime
import base64
import requests
import math
from PIL import ImageFont, ImageDraw
from PIL import Image, ExifTags

import sys
import argparse
import darknet_images
import configparser


config = configparser.ConfigParser()
config_path  = "./config.ini"
config.read(config_path)
# Variable #
queue_name = config.get('rabbitMQ', 'QUEUE_NAME')
diseaseClass = int(config.get('rice-class', 'RICE_CLASS'))

print("Starting")

url = os.environ.get('CLOUDAMQP_URL', 'amqp://admin:Password...@10.99.5.246/%2f')
params = pika.URLParameters(url)

i_class = None
class_num = int(config.get('rice-class', 'RICE_CLASS'))

def RGBgen(diseaseClass):
    colors = []
    for rgb_color in range(diseaseClass):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        rgb_color = (red, green, blue)
        colors.append(rgb_color)
    return colors

def disease(class_id):
    section_name = "rice-disease"
    num_option = len(config.options(section_name))
    rice_dies = []
    for i in range(num_option):
        option = config.options(section_name)[i]
        value = str(config.get(section_name, option))
        rice_dies.append(value)
        if rice_dies[i] == class_id:
            i_class = i
            return i_class
        else:
            i_class = i+1
            return i_class

color_map = RGBgen(diseaseClass)

connection = pika.BlockingConnection(params)
channel = connection.channel()
channel.queue_declare(queue=queue_name)
channel.queue_purge(queue_name)
try:
    darknet_images.performDetect("try")
except:
    print("...")

def draw_rectangle(draw, coordinates, color, width=1):
    #for i in range(width):
    rect_start = (coordinates[0], coordinates[1])
    rect_end = (coordinates[0] + coordinates[2], coordinates[1] + coordinates[3])
    draw.rectangle((rect_start, rect_end), outline = color)

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(class_id)
    iclass = None
    ext_lenght = 50
    iclass = disease(class_id)
    if iclass == diseaseClass:
        all_t = ''
        return all_t
    else:
        color = color_map[iclass]
        cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), color, 8)
        if y < 30:
            cv2.rectangle(img, (x,y), (x+250+ext_lenght, y+40), color, -1)
            cv2.putText(img, label+" "+str((confidence))[:5]+"%", (x+5,y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        else:
            cv2.rectangle(img, (x,y-40), (x+250+ext_lenght,y), color, -1)
            cv2.putText(img, label+" "+str((confidence))[:5]+"%", (x+5,y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        all_t = str(class_id)+" "+str(round(math.floor(confidence),2))+" "+str(x)+" "+str(y)+" "+str(x_plus_w)+" "+str(y_plus_h)+"\n"
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return all_t

def on_request(ch, method, props, body):
    #print(len(body.split(b' ')))
    ai_true = ""
    if len(body.split(b' ')) == 4:
        ai_true,h,w,imgarray = body.split(b' ')
    else:
        h,w,imgarray = body.split(b' ')
    ##print(body)
    
    scale_up_w =  int(h)/416
    scale_up_h =  int(w)/416
    print("**************    recieve     **********************")
    print("Get item at : "+ str(datetime.datetime.now()))
    print("ImgarrayType: ",type(imgarray))
    #print("%.50s" %imgarray)
    try:
        imgarray = np.frombuffer(base64.b64decode(imgarray), np.uint8)
        z = base64.b64decode(imgarray)
    except Exception as e:
        print("Error ==> ",e)
        ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=str("error"))
        ch.basic_ack(delivery_tag = method.delivery_tag)
        return 0
    try:
        imgarray = np.reshape(imgarray, (int(w), int(h), 3))
    except Exception as e:
        imgarray = np.reshape(imgarray, (int(w), int(h), 4))
        imgarray = imgarray[:,:,:3]
        print("Error => ",e)
    print("image array.shape : ",imgarray.shape)
    print(scale_up_h,scale_up_w)
    if len(body.split(b' ')) != 4:
        img_darknet,item_set = darknet_images.performDetect(imgarray)
        print(img_darknet.shape)
        #print("get item : "+str(item_set))
        img = imgarray.copy()
        #drawing = ImageDraw.Draw(im)
        #draw_rectangle(drawing, coordinates, color, width=1)
        all_t = ""
        for i_set in item_set:
            iclass = i_set[0]
            iconfi = i_set[1]
            coor = i_set[2]
            x, y, w_o, h_o = coor
            xmin = int(round(x - (w_o / 2))*scale_up_w)
            xmax = int(round(x + (w_o / 2))*scale_up_w)
            ymin = int(round(y - (h_o / 2))*scale_up_h)
            ymax = int(round(y + (h_o / 2))*scale_up_h)

            if xmin < 0:
                xmin=0
            if ymin < 0:
                ymin=0
            if xmax > int(w):
                xmax= int(w)
            if xmax > int(h):
                xmax= int(h)
            #x,y,w_o,h_o = (int(x)/416)*int(w) , (int(y)/416)*int(h) , (int(w_o)/416)*int(w) , (int(h_o)/416)*int(h)
            #x,y,w_o,h_o = x*scale_up_w , y*scale_up_h , w_o*scale_up_w  , h_o*scale_up_h
            #x_plus_w,y_plus_h = w_o , h_o
            print("iclass...: ",iclass,iconfi,xmin,ymin,xmax,ymax)
            if iclass != disease(iclass):
               all_t = ''
	#condition for dpf class
            if str(iclass) != "dpf_panicle" and float(iconfi) < 50:
                continue
            #all_t = all_t+draw_bounding_box(img, iclass, float(iconfi), int(x-x_plus_w/2), int(y-y_plus_h/2), int(x+x_plus_w/2), int(y+y_plus_h/2))
            all_t = all_t+draw_bounding_box(img, iclass, float(iconfi), int(xmin), int(ymin), int(xmax), int(ymax))

        #print(img.shape)
        #img,item_set = yolo.detect_image(Image.fromarray(imgarray[:,:,::-1]))
        response = np.array(img)
        pack_out = str(w)+' '+str(h)+' '+ str(base64.b64encode(response)) +'-*sep*-'+str(all_t)
        #print(str(w)+' '+str(h)+' '+'-*sep*-'+str(all_t))
    else:
        item_set = darknet_images.performDetect(imgarray[:,:,::-1])
        #w,h = img.size
        img = imgarray[:,:,::-1].copy()
        all_t = ""
        for i_set in item_set:
            iclass = i_set[0]
            iconfi = i_set[1]
            coor = i_set[2]
            x,y,w_o,h_o = coor
            x_plus_w,y_plus_h = x+w_o , y+h_o
            print(iclass,iconfi,x,y, x_plus_w,y_plus_h)
            if str(iclass) != "dpf_panicle" and float(iconfi) < 50:
                continue
            all_t = all_t+draw_bounding_box(img, iclass, iconfi, int(x-x_plus_w/2), int(y-y_plus_h/2), int(x+x_plus_w/2), int(y+y_plus_h/2))
        response = np.array(img[:,:,::-1])
        #url = 'http://messi-fan.org/post'
        #files = {'file': open('image.png', 'rb')}
        #r = requests.post(url, files=files)
        ret, buf = cv2.imencode( '.jpg', cv2.cvtColor(response,cv2.COLOR_BGR2RGB) )
        import requests
        url = 'http://rice-img.openservice.in.th/upload'
        files = {'uploadfile': np.array(buf).tostring() }
        headers = {'content-type': 'image/jpeg;base64'}
        r_im = requests.post(url, files=files)
        #r = "http://example-storage/rice-image/12345.jpg"
        r = r_im.text
        #print(r)
        pack_out = str(r)+'-*sep*-'+str(all_t)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=pack_out)
    ch.basic_ack(delivery_tag = method.delivery_tag)
    print("Done")

# channel.basic_qos(prefetch_count=1)
# channel.basic_consume(queue_name,on_request)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue_name,on_request)
print(" [x] Awaiting RPC requests")
channel.start_consuming()



