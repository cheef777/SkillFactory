#!/usr/bin/env python
import pika
import sys
import time
import pandas as pd
import json

#from Rabbit.union import EXCHANGE1

#login server, type:str
LOGIN = 'guest'
#password server, type:str
PASSWORD = 'guest'
#host server, name or IP adress, type:str
HOST = 'localhost'
#port number, type:int
PORT = 5672
#queue name, type:str
EXCHANGE1 = 'send_in1'
QUEUE = 'send_in1'
df = pd.read_json('g:\RabbitMQ\datasets\dataset.json')
count = 0

while True:
    credentials = pika.PlainCredentials(LOGIN, PASSWORD)
    parameters = pika.ConnectionParameters(host=HOST, port=PORT, credentials=credentials)
    connection = pika.BlockingConnection(parameters)

    channel = connection.channel()
    channel.exchange_declare(exchange=EXCHANGE1)
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.queue_bind(QUEUE, EXCHANGE1, QUEUE)

    code = df['code'][count]
    product = df['product'][count]
    descr = df['fulldescr'][count]
    labels = df.supplier_categories_string[count]
    mess_out = json.dumps({'code': code, 'Product name': product, 'Product description': descr }) 
    
    count +=1
    message = mess_out
    properties=pika.BasicProperties(delivery_mode = 2,) # make message persistent
    channel.basic_publish(exchange='',
                      routing_key=QUEUE,
                      body=message,
                      properties=properties)
    #print( " [x] Sent %r" % (message,))
    print(labels)
    time.sleep(10)
connection.close()