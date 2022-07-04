#!/usr/bin/env python
import pika
import json
import joblib
import pandas as pd
import torch
from datasets import Dataset
from sklearn.neighbors import KNeighborsClassifier
from transformers import set_seed, AutoTokenizer, AutoConfig, DebertaForSequenceClassification


#login server, type:str
LOGIN = 'guest'
#password server, type:str
PASSWORD = 'guest'

#host server, name or IP adress, type:str
HOST = 'localhost'
#port number, type:int
PORT = 5672
VIRTUAL_HOST = '/'

#queue name, type:str
QUEUE1 = 'send_in1'
QUEUE2 = 'send_out1'
PREFETCH_COUNT = 1

#exchange name, type:str
EXCHANGE1 = 'send_in1'
EXCHANGE2 = 'send_out1'
EXCHANGE_TYPE = 'direct'

#pretrained model
PRE_TRAINED_MODEL_NAME = 'microsoft/deberta-large'

RANDOM_STATE = 42
MAX_LEN = 400

#путь хранения файла нейронной сети
output_model_dl_path = ['g:\RabbitMQ\work\model office/', 'g:\RabbitMQ\work\model chewy/']
#путь хранения файла классификатора
output_model_ml_path = ['g:\RabbitMQ\work\model office\model_ml.pkl', 'g:\RabbitMQ\work\model chewy\model_ml.pkl']
#путь хранения файла меток
unique_labels_path = ['g:\RabbitMQ\work\model office\\unique_labels.pkl', 'g:\RabbitMQ\work\model chewy\\unique_labels.pkl']
#список для загрузки нейронных сетей
list_model_dl = []
#список для моделей классификатора
list_model_ml = []
#список для меток моделей
list_unique_labels = []

#список кодов запросов
list_code = ['office', 'chewy']

#установка генератора рандомайзера
set_seed(RANDOM_STATE)

# получение токенов
def tokenize(batch):
    return tokenizer(batch['header'],batch['text'],max_length=MAX_LEN,padding='max_length',truncation='longest_first')

#определиние устройства для нейронных сетей
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# загрузка моделей
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME) #do_lower_case=True
for i in range(len(list_code)):
    config = AutoConfig.from_pretrained(output_model_dl_path[i])
    list_model_dl.append(DebertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=output_model_dl_path[i], config=config))
    list_model_ml.append(joblib.load(output_model_ml_path[i]))
    list_unique_labels.append(joblib.load(unique_labels_path[i]))


# установка канала
credentials = pika.PlainCredentials(LOGIN, PASSWORD)
parameters = pika.ConnectionParameters(host=HOST, 
                                        port=PORT,
                                        virtual_host=VIRTUAL_HOST,
                                        credentials=credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

#линия получения запроса
channel.exchange_declare(exchange=EXCHANGE1, durable=True, exchange_type=EXCHANGE_TYPE)
channel.queue_declare(queue=QUEUE1, durable=True) 
channel.queue_bind(QUEUE1, EXCHANGE1, routing_key=QUEUE1)

#линия отправки ответа
channel.exchange_declare(exchange=EXCHANGE2, exchange_type=EXCHANGE_TYPE, durable=True)
channel.queue_declare(queue=QUEUE2, durable=True)
channel.queue_bind(QUEUE2, EXCHANGE2, routing_key=QUEUE2)


print (' [*] Waiting for messages. To exit press CTRL+C')

#получение запроса
def callback(ch, method, properties, body):
    #print( " [x] Received %r" % (body,))
    
    test_mean = torch.Tensor().to(device)

    # декодировка запроса
    body_dec = body.decode('utf-8')
    message_in = json.loads(body_dec)

    #выбор моделей
    code = list_code.index(message_in['code'])
    model_dl = list_model_dl[code]
    model_ml = list_model_ml[code]
    unique_labels = list_unique_labels[code]

#помещаем нейронную сеть в устройство (GPU or CPU)
    model_dl.to(device)
    
    # предобработка запроса
    texts = pd.Series(message_in['Product description']).to_numpy()
    headers = pd.Series(message_in['Product name']).to_numpy()
    dataset = Dataset.from_dict({'text': texts, 'header': headers})
    dataset = dataset.map(tokenize, batched=True, batch_size=1)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    input_ids = torch.unsqueeze(dataset[0]['input_ids'], 0).to(device)
    attention_mask = torch.unsqueeze(dataset[0]['attention_mask'], 0).to(device)
    output = model_dl(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_layer_number = len(output['hidden_states']) - 1

# векторизация запроса нейронной сетью
    with torch.no_grad():
        output = model_dl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)
    
    mean_output = torch.mean(output['hidden_states'][last_layer_number], dim=1)
    test_mean = torch.cat((test_mean, mean_output), dim=0)

    # получение категоризации из классификатора(KNN)
    preds = model_ml.predict(test_mean.cpu())

# предобработка ответа
    #получение категории из списка меток
    right_category = list(unique_labels[preds])
    message_out = message_in.copy()
    #формируем новое поле с категорией в запросе
    message_out['categorise'] = right_category
    #print(message_out)
    message_out_json = json.dumps(message_out)

    ch.basic_ack(delivery_tag = method.delivery_tag)
    
    # отправка ответа
    properties=pika.BasicProperties(delivery_mode = 2,) # make message persistent
    channel.basic_publish(exchange=EXCHANGE2,
                      routing_key=QUEUE2,
                      body=message_out_json,
                      properties=properties)

channel.basic_qos(prefetch_count=PREFETCH_COUNT)
channel.basic_consume(queue=QUEUE1, on_message_callback=callback)

channel.start_consuming()