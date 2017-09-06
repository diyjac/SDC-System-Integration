### TODO
1. ~~Issue 2. "Velocity should be capped using waypoint_loader and not waypoint_updater"~~
2. ~~Issue 3. "Stopping for Traffic Light not working all that well" (John)~~
3. ~~Issue 5. "If I restart styx when the car is in the middle of the track (not the starting point) then the car doesn't move at all" (Rainer + x)~~
4. ~~Issue 6. "I don't see brake command being sent at all (Except for the first Traffic Light)"~~
9. ~~Resubmit Project~~


### Remarks
##### ref 2. 
I looked again at some of the videos posted before in our channel and saw that sometimes the traffic light when the car is far is not that well recognized.
Idea 1: Use some sort of heatmap (as in the car detection project) to collect detection over several frames and base the decision whether to brake or not on that "averaged" value

##### ref 3.

["Since a safety driver may take control of the car during testing, you should not assume that the car is always following your commands. If a safety driver does take over, your PID controller will mistakenly accumulate error, so you will need to be mindful of DBW status. The DBW status can be found by subscribing to /vehicle/dbw_enabled."](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/877ed434-6955-4371-afcc-ff5b8769f0ce)

1. check whether we take the status of dbw_enabled into account
2. put waypoints cache on the diagscreen and check whether they are still present after styx relaunch?
3. Do we need to reinitialize anything else after dbw_enabled = False?
4. Will a relaunch of styx generate a dbw_enabled = False?

~~delete~~
