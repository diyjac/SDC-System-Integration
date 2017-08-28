# Development and Integration Testing Report

## Safety Assumptions

1. Carla will be tested at less than 10 miles per hours speed.
2. Rosbag image samples are real world samples for classifier training/testing
3. Carla should not go any further than the last waypoint given (no looping).
4. Safety driver will be present when testing with Carla in the field.

## Custom Diagnostic Tools

1. dumpCurrentPos.py
2. dumpCurrentTwist.py
3. dumpWaypoints.py
4. grabFrontCameraImage.py
5. viewFrontCamera.py
6. dumpCurrentPosSteer.py
7. dumpFinalWaypoints.py
8. fakeGreenLight.py
9. diagScreen.py

## Development and Testing

### 1. Performance

#### 1.1 Latency Issues

#### 2.2 Resource Management

### 2. Following Waypoints:

#### 2.1 Waypoint Updater

#### 2.2 Drive By Wire

### 3. Stopping at Traffic Lights

#### 3.1 Traffic Light Waypoint Mapper

#### 3.2 Stopping at All Traffic Lights

Test drive to all intersections in simulator with traffic lights and test if system will stop the car in the simulator.  This test completed successfully as shown below:

##### Traffic Light 0

![./imgs/stop_at_traffic_light_0.jpg](./imgs/stop_at_traffic_light_0.jpg)

##### Traffic Light 1

![./imgs/stop_at_traffic_light_1.jpg](./imgs/stop_at_traffic_light_1.jpg)

##### Traffic Light 2

![./imgs/stop_at_traffic_light_2.jpg](./imgs/stop_at_traffic_light_2.jpg)

##### Traffic Light 3

![./imgs/stop_at_traffic_light_3.jpg](./imgs/stop_at_traffic_light_3.jpg)

##### Traffic Light 4

![./imgs/stop_at_traffic_light_4.jpg](./imgs/stop_at_traffic_light_4.jpg)

##### Traffic Light 5

![./imgs/stop_at_traffic_light_5.jpg](./imgs/stop_at_traffic_light_5.jpg)

##### Traffic Light 6

![./imgs/stop_at_traffic_light_6.jpg](./imgs/stop_at_traffic_light_6.jpg)

##### Traffic Light 7

![./imgs/stop_at_traffic_light_7.jpg](./imgs/stop_at_traffic_light_7.jpg)


#### 3.3 Traffic Light Classifier

#### 3.4 Traffic Light Detector

#### 3.5 Last Waypoint Behavior

As requested by Udacity, the system will stop Carla at the last waypoint, as shown below:

![./imgs/last_waypoint.jpg](./imgs/last_waypoint.jpg)
