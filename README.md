# **Squats-Tracker**
Squats-Tracker is a **Flask** Application which is used track and count the number of squats in real-time using **OpenCV** in **python**.
### ***[`DEMO`](https://youtu.be/60kxRnQEW94)***

## Features
* Detects Human pose using pre-trained *COCO model*.
* Calculates angle between thigh and calf for every frame and increments the count whenever the angle gets less than 100&deg;.
* Angle calculated between thigh and calf is streamed to the **client side** in *real-time*.
* Number of squats are stored for every session with date and time for every user using `SQLite` database.
* Previously recorded squats are represented through bar-chart using `Chart.js`.

## How to install and run the project
* Use command line to clone project on your machine.
`git clone https://github.com/mukul-rana/Squats-Tracker.git`
* Install python using `anaconda` which already contains most of the python modules, otherwise install each module using `pip`.
* To use Nvidia GPU (for smooth and fast processing), you need to install `CUDA` and `cuDNN`.
* Download [pose_deploy_linevec.prototxt](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt) and [pose_iter_440000.caffemodel](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/raw/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel) files and place in `app/static` folder.
* Open windows power-shell in Squats-Tracker folder and setup flask app root folder by using command : ` $env:FLASK_APP = "app"`
> For bash or cmd, please refer : https://flask.palletsprojects.com/en/2.0.x/tutorial/factory/#run-the-application
* Use `flask run` to start the server.
* Server starts running on port 5000, you can access by typing `localhost:5000` in a browser window.