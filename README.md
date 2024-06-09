# Tello Drone PI Controller

This project demonstrates a PI (Proportional-Integral) controller using a Tello drone to follow a person using object detection. The detection is performed using a pre-trained MobileNet SSD neural network model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tello-drone-pi-controller.git
   cd tello-drone-pi-controller


2. Download the pretrained models..
   wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt -O models/MobileNetSSD_deploy.prototxt.txt
   wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel -O models/MobileNetSSD_deploy.caffemodel

3. install the dependencies.
   pip install -r requirements.txt

Usage...

run the script 
  python src/drone_controller.py
  The drone will take off and attempt to follow any person it detects within the frame using the PI controller.

Press q to land the drone and stop the script
