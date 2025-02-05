import rclpy
from std_msgs.msg import String
import sys

def main(args=None):
    rclpy.init(args= args)
    # Creating Our Node 
    node = rclpy.create_node("Send_Dir")
    #Creating Publisher Of Our Node
    '''This publisher send a string carrying the character that will control 
    the direction & speed of the delivery car
    w --> move forward
    s --> move backward
    a --> move left
    d --> move right
    z --> stop movement
    1 --> first speed
    2 --> second speed
    3 --> third speed
    4 --> fourth speed
    5 --> fifth speed'''
    publisher= node.create_publisher(String,"topic",10)
    msg =String()

    def timer_callback():
        msg.data = "'" + sys.argv[1] + "'"
        node.get_logger().info("%s"% msg.data)
        publisher.publish(msg)
    
    timer = node.create_timer(0.5,timer_callback)
    
    rclpy.spin(node)
    node.destroy_timer(timer)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    
