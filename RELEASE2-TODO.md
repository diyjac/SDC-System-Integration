### TODO
1. Issue 2. "Velocity should be capped using waypoint_loader and not waypoint_updater"
2. Issue 3. "Stopping for Traffic Light not working all that well"
3. Issue 5. "If I restart styx when the car is in the middle of the track (not the starting point) then the car doesn't move at all"
4. Issue 6. "I don't see brake command being sent at all (Except for the first Traffic Light)"
9. Resubmit Project


### Remarks
##### ref 2. 
I looked again at some of the videos posted before in our channel and saw that sometimes the traffic light when the car is far is not that well recognized.
Idea 1: Use some sort of heatmap (as in the car detection project) to collect detection over several frames and base the decision whether to brake or not on that "averaged" value

~~delete~~