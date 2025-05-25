# Práctica 2: Carrera de Robots con TurtleBot3

Este proyecto forma parte de la asignatura **Visión Artificial y Robótica** y consiste en una simulación de una carrera de robots utilizando TurtleBot3 en un entorno de múltiples robots.

## Objetivo

El objetivo de la práctica es diseñar un sistema de navegación autónoma para que varios robots TurtleBot3 compitan en una carrera. La solución puede incluir algoritmos de planificación, detección de obstáculos, seguimiento de trayectorias y/o uso de sensores.

## Lanzar el entorno de simulación

Antes de ejecutar la simulación, asegúrate de estar en el directorio raíz de tu espacio de trabajo (`catkin_ws` o equivalente), y ejecuta los siguientes comandos en la terminal:

```bash
export TURTLEBOT3_MODEL=waffle
source devel/setup.bash
roslaunch turtlebot_gazebo_multiple create_multi_robot.launch
```

## Teleoperar el robot

Para teleoperar el robot y poder conseguir más imágenes para el dataset debemos lanzar los siguientes comandos en otra terminal:

```bash
export TURTLEBOT3_MODEL=waffle
source devel/setup.bash
rosrun turtlebot3_teleop turtlebot3_teleop_key
```

El único requisito es tener el paquete de turtlebot3.

## Grabar imágenes nuevas para el dataset

Solo hay que hacer uso en otra terminal distinta de la herramienta incluida en el paquete navegacion_autonoma con el siguiente comando:

```bash
export TURTLEBOT3_MODEL=waffle
source devel/setup.bash
rosrun navegacion_autonoma grabar_dataset.py
```

Y cuando ya no queramos grabar más imágenes cerrarlo con Ctrl + C.

## Entrenar un nuevo modelo

Se puede entrenar un nuevo modelo una vez capturadas más imágenes, esto se hace a través de este comando en otra terminal:

```bash
cd src/navegacion_autonoma
python3 src/entrenar_modelo_cnn.py
```

Esto generará un nuevo modelo dentro de la carpeta 

```bash
/models/modelo_entrenado.h5
```

Solo hay que sustituir el modelo antiguo por este nuevo que se cree. El modelo entrenado debe tener el nombre especificado.

## Lanzar el nodo de navegación

Para lanzar el nodo de navegación debemos poner los siguientes comandos en otra terminal:

```bash
export TURTLEBOT3_MODEL=waffle
source devel/setup.bash
roslaunch navegacion_autonoma navegacion.launch
```



