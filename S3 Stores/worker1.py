#!/usr/bin/env python
import pika
import json

#from Rabbit.union import EXCHANGE2

#login server, type:str
LOGIN = 'guest'
#password server, type:str
PASSWORD = 'guest'
#host server, name or IP adress, type:str
HOST = 'localhost'
#port number, type:int
PORT = 5672
#queue name, type:str
EXCHANGE2 = 'send_out1'
QUEUE = 'send_out1'
VIRTUAL_HOST = '/'

credentials = pika.PlainCredentials(LOGIN, PASSWORD)
parameters = pika.ConnectionParameters(host=HOST, 
                                        port=PORT,
                                        virtual_host=VIRTUAL_HOST,
                                        credentials=credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.exchange_declare(EXCHANGE2)#, exchange_type='direct', durable=True)
channel.queue_declare(queue=QUEUE, durable=True)
channel.queue_bind(QUEUE, EXCHANGE2, QUEUE)

print (' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):
    body_dec = body.decode('utf-8')
    message_in = json.loads(body_dec)
    print( message_in['categorise'])
   
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=QUEUE, on_message_callback=callback)

channel.start_consuming()