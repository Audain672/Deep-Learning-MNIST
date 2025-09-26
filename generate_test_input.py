import tensorflow as tf
import json
import numpy as np

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

IMAGE_INDEX = 0
single_image = x_test[IMAGE_INDEX]
true_label = y_test[IMAGE_INDEX]

print(f"L'image sélectionnée est un : {true_label}")
print("-" * 30)

flattened_image = single_image.flatten()

image_as_list = flattened_image.tolist()

json_payload = {
    "image": image_as_list
}

print("Payload JSON à envoyer à l'API :")
print(json.dumps(json_payload))
print("-" * 30)