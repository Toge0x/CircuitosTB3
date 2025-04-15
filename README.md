# Práctica: Carrera de Robots con TurtleBot3

Este proyecto forma parte de la asignatura **Visión Artificial y Robótica** y consiste en una simulación de una carrera de robots utilizando TurtleBot3 en un entorno de múltiples robots.

## 🏁 Objetivo

El objetivo de la práctica es diseñar un sistema de navegación autónoma para que varios robots TurtleBot3 compitan en una carrera. La solución puede incluir algoritmos de planificación, detección de obstáculos, seguimiento de trayectorias y/o uso de sensores.

## 🧰 Requisitos

- ROS Noetic
- Gazebo
- Paquetes de TurtleBot3 instalados correctamente
- Dependencias del entorno múltiple configuradas

## 🚀 Lanzar el entorno de simulación

Antes de ejecutar la simulación, asegúrate de estar en el directorio raíz de tu espacio de trabajo (`catkin_ws` o equivalente), y ejecuta los siguientes comandos en la terminal:

```bash
export TURTLEBOT3_MODEL=waffle
source devel/setup.bash
roslaunch turtlebot_gazebo_multiple create_multi_robot.launch

