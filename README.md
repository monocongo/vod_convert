# vod_convert
Python modules and scripts for conversion between popular visual object detection annotation formats.

## Install the TensorFlow Object Detection API
Since we'll utilize the TensorFlow models API in order to perform TFRecord 
conversions we'll need to have access to the 
[TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

(**NOTE**: the below is a simplification of the 
[official installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))

1. Clone the TensorFlow Object Detection API from GitHub and set the API's base
directory as an environment variable for later use:
    ```bash
    $ cd <directory_of_your_choice>
    $ git clone git@github.com:tensorflow/models.git
    $ cd models
    $ export TFOD=`pwd`
    ```
2. Create a new Python environment (or activate an existing Python environment).
In this example we'll use an Anaconda environment:
    ```bash
    $ conda create -n tltod python=3 --yes
    $ conda activate tltod
    ```
3. Install TensorFlow and other necessary packages:
    ```bash
    $ pip install tensorflow
    $ pip install Cython
    $ pip install contextlib2
    $ pip install pillow
    $ pip install lxml
    $ pip install jupyter
    $ pip install matplotlib
    ```
4. Compile the Protobuf libraries:
    ```bash
    $ cd ${TFOD}/research
    $ protoc object_detection/protos/*.proto --python_out=.
    ```
5. Add the API's `research` and `research/slim` directories to the `PYTHONPATH` variable:
    ```bash
    $ cd ${TFOD}/research
    $ export PYTHONPATH=`pwd`:`pwd`/slim
    ```
6. Test the installation:
    ```bash
    $ python object_detection/builders/model_builder_test.py
    ```

