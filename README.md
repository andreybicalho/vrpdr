# Deep Learning Applied To Vehicle Registration Plate Detection and Recognition

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

# What's this repo about?

This is a simple approach for vehicle registration plate detection and recognition. It is not an end-to-end system, instead, two different deep learning methods were stacked together to complete this task. [*YOLO*](https://github.com/AlexeyAB/darknet) object detection algorithm was used to detect license plate regions, then an `Attention Based Optical Character Recognition` [*Attention-OCR*](https://github.com/wptoux/attention-ocr) was applied to recognize the characters.

![Output](docs/result.jpg "Output")*Results (vehicle license plate and recognized characters were intentionally blurred).*

# Install and Requirements

````
pip install -r requirements.txt
````

## Pre-trained Weights

Download the pre-trained weights for the YOLO and the Attention-OCR and put it in the `config` directory.

* *YOLO* and *Attention-OCR* were trained on the Brazilian [SSIG-ALPR](http://smartsenselab.dcc.ufmg.br/en/dataset/banco-de-dados-sense-alpr/) dataset.
  * `TODO:` upload weights and other config files somewhere.

# Running

Run the application API:
````
python app.py
````

The app will be listening to requests on http://localhost:5000/

Send an Http POST request with a form-data body with an attribute `file` containing the image, like this:

````
curl --location --request POST 'localhost:5000/' \
--form 'file=@/path/to/the/image/file/image_file.png'
````

### API Output:

The API will output all the detections with the corresponding bounding boxes and its confidence scores as well as the OCR prediction for each bounding box. Also, we draw all these information on the input image and outputs it as a base64 image.

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

*Note: If `DEBUG` flag is set to `True` in the `app.py`, images will be produced in the `debug` directory to make debug a bit easier.*
