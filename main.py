#!/usr/bin/env python
import io
from pathlib import Path
from google.cloud import vision
from google.cloud.vision import types


def detect_faces(file_path):
    client = vision.ImageAnnotatorClient()
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    for face in faces:
        print('\tAnger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('\tJoy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('\tSurprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        # vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices])
        # print('\tFace bounds: {}'.format(','.join(vertices)))
    if response.error.message:
        exit(response.error.message)


def detect_labels(file_path):
    client = vision.ImageAnnotatorClient()
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        print("\t{}".format(label.description))


# export GOOGLE_APPLICATION_CREDENTIALS="/home/op/Downloads/credentials.json"
if __name__ == "__main__":
    path = Path("/tmp/2020-02-27-205507.jpg")
    detect_faces(path)
    detect_labels(path)
