# Deep Learning Applied To Vehicle Registration Plate Detection and Recognition

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

# What's this repo about?

This is a simple approach for handling the problem of vehicle license plate recognition. It is not an end-to-end system, instead, two different deep learning algorithms were stacked together to complete this task. First, license plates regions were extracted by using the [*YOLO*](https://github.com/AlexeyAB/darknet) object detection algorithm, then the region proposals were submitted to an `Attention Based Optical Character Recognition`, [*Attention-OCR*](https://github.com/wptoux/attention-ocr), to finally recognize the characters.

![](docs/example.gif)

# Running

Make sure you have all the dependencies installed:

````
pip install -r requirements.txt
````

Both *YOLO* and *Attention-OCR* were trained on the Brazilian [SSIG-ALPR](http://smartsenselab.dcc.ufmg.br/en/dataset/banco-de-dados-sense-alpr/) dataset:

* Images were resized to 1056x576 during training, so YOLO will perform best if applied to this shape.
* Cropped bounding box images (i.e. license plates) were resized to 160x60 to train the Attention-OCR.

[Download](https://drive.google.com/drive/folders/1Ug2UpsQ7tfcDVIW6P3UQ3uUfUnNmNLy-?usp=sharing) the pretrained models as well as the configuration files and put them in the `config` directory. 

Run the application service:
````
python app.py
````

The application service will be listening to requests on http://localhost:5000/

Send an Http POST request with a form-data body with an attribute `file` containing the image, like this `curl` example:

````
curl --location --request POST 'localhost:5000/' \
--form 'file=@/path/to/the/image/file/image_file.png'
````

# Output

The API will output all the detections with the corresponding bounding boxes and its confidence scores as well as the OCR prediction for each bounding box. Also, we draw all these information on the input image and also outputs it as a base64 image.

`json object` response will look like the following:

````
{
  "detections": [
    {
      "bb_confidence": 0.973590612411499,
      "bounding_box": [
        1509,
        877,
        82,
        39
      ],
      "ocr_pred": "ABC1234-"
    },
    {
      "bb_confidence": 0.9556514024734497,
      "bounding_box": [
        161,
        866,
        100,
        40
      ],
      "ocr_pred": "ABC1234-"
    }
  ],
  "output_image": "/9j/4AAQS..."
}
````

*Note: If `DEBUG` flag is set to `True` in the `app.py`, images will be produced in the `debug` directory to make debugging a bit easier.*
