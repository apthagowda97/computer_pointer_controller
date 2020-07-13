# Computer Pointer Controller

This project helps in controlling the mouse pointer in realtime through the direction of face and eyes using Intel's Distribution of OpenVino.


## Project Set-Up and Installation

### System Details:
```
OpenVino : openvino_2020.1.033
Hardware : Intel i5 5th gen CPU 
OS       : Windows 
Models   : 1.face-detection-adas-binary-0001
           2.landmarks-regression-retail-0009
           3.head-pose-estimation-adas-0001
           4.gaze-estimation-adas-0002
```
### Directory structue:
```
├───bin
|       demo.mp4
├───model
│   └───intel
│       ├───face-detection-adas-binary-0001
│       │   └───FP32-INT1
│       ├───gaze-estimation-adas-0002
│       │   ├───FP16
│       │   ├───FP32
│       │   └───FP32-INT8
│       ├───head-pose-estimation-adas-0001
│       │   ├───FP16
│       │   ├───FP32
│       │   └───FP32-INT8
│       └───landmarks-regression-retail-0009
│           ├───FP16
│           ├───FP32
│           └───FP32-INT8
└───src
        computer_pointer.ipynb
        face_detection.py
        gaze_estimation.py
        head_pose.py
        input_feeder.py
        landmark_regression.py
        main.py
        mouse_controller.py
        result.txt

```
*The models are not included in the zip file. Download the above models in the `models` directory as shown in the directory tree using `model_downloder`.*

## Demo

To run the demo just fire `python main.py`. It will internally take the `FP32` precision of all the models with a video file(video mode) as default.

To run custom `main.py`:
```
python main.py -f [dir_path]\model\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 \
 -he [dir_path]\model\intel\head-pose-estimation-adas-0001\FP32-INT8\head-pose-estimation-adas-0001 \
 -l [dir_path]\model\intel\landmarks-regression-retail-0009\FP32-INT8\landmarks-regression-retail-0009 \
 -g [dir_path]\model\intel\gaze-estimation-adas-0002\FP32-INT8\gaze-estimation-adas-0002 \ 
 -t video \
 -i [dir_path]\bin\demo.mp4 
```
*Where `dir_path` is the path the abs path.*

## Documentation

the `-help` for `main.py`:

```
usage: main.py [-h] [-f FACE_MODEL] [-he HEAD_MODEL] [-l LANDMARK_MODEL]
               [-g GAZE_MODEL] [-t INPUT_TYPE] [-i INPUT_FILE] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -f FACE_MODEL, --face_model FACE_MODEL
                        Path to an xml file with a trained model.
  -he HEAD_MODEL, --head_model HEAD_MODEL
                        Path to an xml file with a trained model.
  -l LANDMARK_MODEL, --landmark_model LANDMARK_MODEL
                        Path to an xml file with a trained model.
  -g GAZE_MODEL, --gaze_model GAZE_MODEL
                        Path to an xml file with a trained model.
  -t INPUT_TYPE, --input_type INPUT_TYPE
                        'video', 'cam' or 'image' input type
  -i INPUT_FILE, --input_file INPUT_FILE
                        Path to image or video file
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on
```

## Benchmarks

After running the `main.py`, the result will update in the [result.log](src\result.log) file. It will tell about the model's load time along with the inference time per frame for each model and the total inference time.

| Precision | Model's load time|face infrence|landmark infrence|head infrence|gaze infrence| Avg infrence|
| --- | --- | --- |--- | --- | --- | --- |
| FP32-INT8| 2.7498s |0.0474s | 0.0005s |0.0016s| 0.0066s | 0.0561s |
| FP16| 0.8289s | - | 0.0005s | 0.0034s | 0.0095s | 0.0606s |
| FP32| 0.7979s| - | 0.0011s | 0.0021s | 0.0093s | 0.0596s |


## Results

1. The model load time of `FP32-INT8` is larger than `FP16` and `FP32 `models.
2. The average inference time of `FP32-INT8` is lesser compared with `FP32` which has lesser than `FP16` models.