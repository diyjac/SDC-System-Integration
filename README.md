This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Installation 

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop). 
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space
  
  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
  * [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
    __NOTE__: *If you are installing in native Ubuntu 16.04, the Dataspeed DBW One Line SDK binary install will auto install 4.4.0-92-generic Linux kernel, which will break CUDA and the NVIDIA 375 drivers if you have NVIDIA GPU in your native Ubuntu 16.04 build.  This will cause starting the simulator to fail because it can no longer open the OpenGL drivers provided by NVIDIA:*

    ![starting simulator failure image](./imgs/sim_startup_failure_caused_by_4.4.0-92-generic_kernel.png)

    *To fix this issue, you will have to:*
    * remove the 4.4.0-92-generic kernel (or any kernel newer than 4.4.0-91-generic):
        ```bash
        sudo apt-get remove linux-image-4.4.0-92-generic
        ```
    * reinstall the NVIDIA 375 drivers (follow the instructions):

        [https://askubuntu.com/questions/760934/graphics-issues-after-while-installing-ubuntu-16-04-16-10-with-nvidia-graphics](https://askubuntu.com/questions/760934/graphics-issues-after-while-installing-ubuntu-16-04-16-10-with-nvidia-graphics)
    
* Download the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/v0.1).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/diyjac/SDC-System-Integration.git
```

2. Install python dependencies
```bash
cd SDC-System-Integration
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.bash
roslaunch launch/styx.launch
```
4. Run the simulator
```bash
unzip lights_no_cars.zip
cd lights_no_cars
chmod +x ros_test.x86_64
./ros_test.x86_64
```
5. To test grab a raw camera image
```bash
rosrun tools grabFrontCameraImage.py ../imgs/sampleout.jpg
```
![./imgs/sampleout.jpg](./imgs/sampleout.jpg)

6. To dump the waypoints from the `/base_waypoints` topic
```bash
rosrun tools dumpWaypoints.py ../data/simulator_waypoints.csv
```
7. To dump the final waypoints from the `/final_waypoints` topic
```bash
rosrun tools dumpFinalWaypoints.py ../data/final_waypoints.csv
```

![./imgs/sim_waypoint_map.png](./imgs/sim_waypoint_map.png)

