import pika

ds = "e"
I = 2
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

body = f'{ds}, {I}'
channel.basic_publish(exchange='', routing_key='hello', body=body)
print(f" [x] Sent '{body}!'")
connection.close()