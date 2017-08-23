#include "waypoint_updater.h"

constexpr int LOOP_RATE = 30; //processing frequency


int main(int argc, char **argv)
{


  // set up ros
  ros::init(argc, argv, "waypoint_updater");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  bool linear_interpolate_mode;
  private_nh.param("linear_interpolate_mode", linear_interpolate_mode, bool(true));
  ROS_INFO_STREAM("linear_interpolate_mode : " << linear_interpolate_mode);

  waypoint_follower::PurePursuit pp(linear_interpolate_mode);

  ROS_INFO("set publisher...");
  // publish topic
  ros::Publisher cmd_velocity_publisher = nh.advertise<geometry_msgs::TwistStamped>("twist_cmd", 10);

  ROS_INFO("set subscriber...");
  // subscribe topic
  ros::Subscriber waypoint_subscriber =
      nh.subscribe("final_waypoints", 10, &waypoint_follower::PurePursuit::callbackFromWayPoints, &pp);
  ros::Subscriber ndt_subscriber =
      nh.subscribe("current_pose", 10, &waypoint_follower::PurePursuit::callbackFromCurrentPose, &pp);
  ros::Subscriber est_twist_subscriber =
      nh.subscribe("current_velocity", 10, &waypoint_follower::PurePursuit::callbackFromCurrentVelocity, &pp);

  ROS_INFO("pure pursuit start");
  ros::Rate loop_rate(LOOP_RATE);
  while (ros::ok())
  {
    ros::spinOnce();
    cmd_velocity_publisher.publish(pp.go());
    loop_rate.sleep();
  }

  return 0;
}
