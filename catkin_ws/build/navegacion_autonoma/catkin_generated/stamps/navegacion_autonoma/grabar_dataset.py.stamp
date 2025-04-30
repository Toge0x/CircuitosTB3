#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import csv

class DatasetRecorder:
    def __init__(self):
        rospy.init_node('grabar_dataset', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)

        self.last_velocity = (0.0, 0.0)
        self.counter = 0

        # Crear carpetas para dataset
        self.dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset')
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

        # Crear y abrir el archivo CSV
        self.csv_path = os.path.join(self.dataset_dir, 'labels.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['filename', 'v', 'w'])

        rospy.loginfo("Grabador de dataset iniciado.")

    def vel_callback(self, msg):
        self.last_velocity = (msg.linear.x, msg.angular.z)

    def image_callback(self, msg):
        try:
            # Convertir imagen ROS a OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = f"{self.counter:05d}.png"
            filepath = os.path.join(self.images_dir, filename)

            # Guardar imagen
            cv2.imwrite(filepath, cv_image)

            # Guardar velocidad asociada
            v, w = self.last_velocity
            self.csv_writer.writerow([filename, v, w])
            self.counter += 1

            rospy.loginfo(f"[{self.counter}] Guardada {filename} con v={v:.2f}, w={w:.2f}")

        except Exception as e:
            rospy.logerr(f"Error al procesar imagen: {e}")

    def run(self):
        rospy.spin()
        self.csv_file.close()

if __name__ == '__main__':
    recorder = DatasetRecorder()
    recorder.run()
