#!/usr/bin/env python3

import rospy
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from tensorflow.keras.models import load_model
import rospkg

class Navegacion:
    def __init__(self):
        rospy.init_node('nodo_navegacion', anonymous=True)

        # Cargar modelo desde la ruta del paquete
        rospack = rospkg.RosPack()
        model_path = os.path.join(rospack.get_path('navegacion_autonoma'), 'models', 'modelo_entrenado.h5')
        self.model = load_model(model_path)
        rospy.loginfo("Modelo cargado desde: " + model_path)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.img_size = (64, 64)
        rospy.loginfo("Nodo de navegación iniciado.")

    def preprocess_image(self, cv_image):
        img_resized = cv2.resize(cv_image, self.img_size)
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)  # [1, 64, 64, 3]
        return img_expanded

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.preprocess_image(cv_image)

            prediction = self.model.predict(input_tensor)[0]
            v, w = prediction[0], prediction[1]

            twist = Twist()
            twist.linear.x = max(0.0, v)  # no retroceso
            twist.angular.z = w
            self.cmd_vel_pub.publish(twist)

        except Exception as e:
            rospy.logerr("Error en image_callback: %s", str(e))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        navigator = Navegacion()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
