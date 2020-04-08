import pika

for i in range(10):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='map')

    body = f"{i}"
    channel.basic_publish(exchange='', routing_key='map', body=body)
    print(f" [x] Sent '{body}!'")
    connection.close()