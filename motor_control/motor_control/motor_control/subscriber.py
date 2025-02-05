import rclpy
import serial
from std_msgs.msg import String

ser = serial.Serial(port='/dev/ttyUSB0',baudrate=9600)
node = None
def chatter_callback(msg):
    global node
    node.get_logger().info("%s" % msg.data)
    value = ser.write(bytes(msg.data,'utf-8'))
def main(args = None):
    global node
    rclpy.init(args = args)

    node = rclpy.create_node("Send_Dir")
    node.create_subscription(String,'topic',chatter_callback,10)

    while rclpy.ok():
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
