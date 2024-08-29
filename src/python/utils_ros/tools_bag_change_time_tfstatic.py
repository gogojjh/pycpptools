import rosbag
from tf2_msgs.msg import TFMessage
import rospy
import argparse

def change_timestamp_in_tf_static(input_bag, output_bag, new_timestamp):
    with rosbag.Bag(output_bag, 'w') as outbag:
        with rosbag.Bag(input_bag, 'r') as inbag:
            for topic, msg, t in inbag.read_messages():
                if topic == "/tf_static" and isinstance(msg, TFMessage):
                    # Modify the timestamp of all transforms in /tf_static
                    for transform in msg.transforms:
                        transform.header.stamp = rospy.Time.from_sec(new_timestamp)
                    # Write the modified message to the output bag
                    outbag.write(topic, msg, rospy.Time.from_sec(new_timestamp))
                else:
                    # For all other messages, write them unchanged
                    outbag.write(topic, msg, rospy.Time.from_sec(new_timestamp))

def main():
    parser = argparse.ArgumentParser(description="Change the timestamp of /tf_static messages in a ROS bag file.")
    parser.add_argument("--input_bag", type=str, help="Path to the input ROS bag file.")
    parser.add_argument("--output_bag", type=str, help="Path to save the output ROS bag file.")
    parser.add_argument("--new_timestamp", type=float, help="New timestamp (in seconds since epoch) to set for /tf_static messages.")
    args = parser.parse_args()

    change_timestamp_in_tf_static(args.input_bag, args.output_bag, args.new_timestamp)
    print(f"Modified /tf_static timestamps and saved to {args.output_bag}")

if __name__ == "__main__":
    main()
