#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/foreach.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <geometry_msgs/Twist.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>         // Generalized ICP
#include <pcl/registration/icp_nl.h>       // Non-linear ICP
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/PointIndices.h>
#include <pcl/correspondence.h>
#include <boost/filesystem.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <boost/foreach.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <geometry_msgs/Twist.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>

using namespace pcl;
using namespace std;

namespace fs = boost::filesystem;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr visu_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
ros::Publisher cmd_vel_pub;
int angle_step = 20; // Grados por paso
int total_rotation = 360; // Rotación completa

double angular_speed = 3; //(2 * M_PI) / (total_rotation / angle_step); // Velocidad angular (AJUSTABLE)
const double STEP_DURATION = 2; // Duración del paso en segundos
int steps_left = 20; 

boost::mutex cloud_mutex;

#define USE_HARRIS true // Cambiar a false para usar ISS (true para usar HARRIS)
#define USE_SHOT false // Cambiar a false para usar FPFH (true para usar SHOT)

#define USE_TAKEN_POINTS true // usar imágenes capturadas previamente (FALSE PARA CAPTURAR NUBES NUEVAS)

pcl::PointCloud<pcl::FPFHSignature33>::Ptr prev_desc_FPFH;
pcl::PointCloud<pcl::SHOT352>::Ptr prev_desc_SHOT;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr prev_keypoints;

Eigen::Matrix4f transformation_acumulada = Eigen::Matrix4f::Identity();
pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZRGB>);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// FUNCIONES AUXILIARES //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Función para visualizar las nubes de puntos
void simpleVis()
{
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    while (!viewer.wasStopped())
    {
        boost::mutex::scoped_lock lock(cloud_mutex);
        viewer.showCloud(visu_pc);
        lock.unlock();
        boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    }
}

// Función para calcular las normales de la nube de puntos
pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(120);
    ne.compute(*normals);
    
    return normals;
}

// Función para quitar los puntos NaN
pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeNaNPoints(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);
    
    ROS_INFO("Puntos después de quitar NaN: %lu (eliminados %lu puntos)", 
             cloud_filtered->size(), cloud->size() - cloud_filtered->size());
    
    return cloud_filtered;
}

// Función para calculo dinamico de la resolución media de una nube de puntos para ISS (distancia media al vecino más cercano para cada punto.)
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud) {
    double resolution = 0.0;
    int points = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> sqr_distances(2);
    pcl::search::KdTree<pcl::PointXYZRGB> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (!std::isfinite(cloud->points[i].x)) continue;

        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
        if (nres == 2) {
            resolution += std::sqrt(sqr_distances[1]);
            ++points;
        }
    }

    if (points != 0)
        resolution /= points;

    return resolution;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// DETECTORES ///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Función para detectar keypoints (puntos clave) en una nube de puntos RGB. 
// Devuelve una nueva nube con los keypoints y actualiza por referencia los índices (para conservar las normales) de los keypoints detectados.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr detect_keypoints(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                                        pcl::PointIndicesConstPtr &keypoints_index) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());

    if (USE_HARRIS) {
            // Detector Harris mejorado
        pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI> detector;
        pcl::PointCloud<pcl::PointXYZI>::Ptr harris_response(new pcl::PointCloud<pcl::PointXYZI>);

        detector.setInputCloud(cloud);
        detector.setNonMaxSupression(true);          
        detector.setThreshold(1e-7);                 
        detector.setRadius(0.07);                    
        detector.setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::HARRIS);
        detector.compute(*harris_response);

        // Convertir a XYZRGB
        pcl::copyPointCloud(*harris_response, *keypoints);

        // Obtener índices reales en la nube original
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(cloud);
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        for (const auto& pt : keypoints->points) {
            std::vector<int> nearest(1);
            std::vector<float> distances(1);
            if (kdtree.nearestKSearch(pt, 1, nearest, distances) > 0) {
                indices->indices.push_back(nearest[0]);
            }
        }
        keypoints_index = indices;

        // Validar antes de usar los índices
        if (!keypoints_index || keypoints_index->indices.empty()) {
            ROS_WARN("No se detectaron índices de keypoints válidos. Se omite este paso.");
            return keypoints;
        }

        // Reemplazar keypoints por los puntos correctos de la nube original
        pcl::copyPointCloud(*cloud, keypoints_index->indices, *keypoints);

        ROS_INFO("Keypoints detectados con HARRIS: %lu", keypoints->size());
    } else { // DETECTOR ISS
        pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());     

        double model_resolution = computeCloudResolution(cloud); // Calculo de la resolución del modelo para adaptar los parámetros del detector
        ROS_INFO("Model resolution: %f", model_resolution);
        
        iss.setSearchMethod(tree);
        iss.setInputCloud(cloud);
        
        iss.setSalientRadius(2 * model_resolution); // Radio para detectar características salientes
        iss.setNonMaxRadius(1.5 * model_resolution); // Radio para supresión de no-máximos
        iss.setThreshold21(0.975); // Umbrales para la detección basada en valores propios
        iss.setThreshold32(0.975);
        iss.setMinNeighbors(5); // Mínimo número de vecinos requeridos para considerar el punto
        iss.setNumberOfThreads(4); // Número de hilos para paralelizar el proceso
        
        try {
            iss.compute(*keypoints); // Deteccion de keypoints
            keypoints_index = iss.getKeypointsIndices(); // Se obtienen los índices de los keypoints detectados
            
            ROS_INFO("Keypoints detected: %lu", keypoints->size());
        }
        catch (const std::exception& e) {
            ROS_ERROR("Error in keypoint detection: %s", e.what());
        }
    }

    return keypoints;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// DESCRIPTORES //////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// DESCRIPTOR FPFH
pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_descriptor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                                                    PointCloud<Normal>::Ptr normals,
                                                                    PointIndicesConstPtr& keypoints_index) {
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>);
    
    pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());

    fpfh.setInputCloud(cloud); // Se establece la nube de entrada
    fpfh.setInputNormals(normals); // Se establecen las normales precomputadas para cada punto
    fpfh.setIndices(keypoints_index); // Se establecen los índices de los keypoints sobre los cuales calcular los descriptores
    fpfh.setKSearch(160); // Número de vecinos a considerar (150+- mejor resultado)
    fpfh.setSearchMethod(tree); // Bucar vecinos
    
    fpfh.compute(*descriptors); // Calculo de descriptores
    ROS_INFO("Descriptores FPFH calculados: %lu", descriptors->size());
    
    return descriptors;
}

// DESCRIPTOR SHOT
pcl::PointCloud<pcl::SHOT352>::Ptr compute_shot_descriptor(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
    PointCloud<Normal>::Ptr normals,
    PointIndicesConstPtr& keypoint_index) 
{
    pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>);
    
    if (!cloud || cloud->empty() || !normals || normals->empty() || !keypoint_index) {
        ROS_WARN("Invalid input for SHOT descriptor computation");
        return descriptors;
    }

    try {
        pcl::SHOTEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> shot;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        
        shot.setInputCloud(cloud); // Se establece la nube de entrada
        shot.setInputNormals(normals); // Se establecen las normales precomputadas para cada punto
        shot.setIndices(keypoint_index); // Se establecen los índices de los keypoints sobre los cuales calcular los descriptores
        shot.setSearchMethod(tree); // Bucar vecinos
        shot.setRadiusSearch(0.15); // Se establece el radio de búsqueda
        
        shot.compute(*descriptors); // Calculo de los descriptores
        
        // Filtramos los descriptores inválidos que contienen valores no finitos (NaN o Inf) para evitar el fallo
        pcl::PointCloud<pcl::SHOT352>::Ptr valid_descriptors(new pcl::PointCloud<pcl::SHOT352>);
        for (size_t i = 0; i < descriptors->size(); ++i) {
            const pcl::SHOT352& descriptor = descriptors->points[i];
            bool is_valid = true;
            for (int j = 0; j < 352; ++j) {
                if (!std::isfinite(descriptor.descriptor[j])) {
                    is_valid = false;
                    break;
                }
            }
            if (is_valid) {
                valid_descriptors->push_back(descriptor);
            }
        }
        
        descriptors = valid_descriptors;
        
        ROS_INFO("Valid SHOT descriptors: %lu", descriptors->size());
    } 
    catch (const std::exception& e) {
        ROS_ERROR("Exception in SHOT descriptor computation: %s", e.what());
    }
    
    return descriptors;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// EMPAREJAMIENTO (BUSCAR CORRESPONDENCIAS) ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Función para encontrar correspondencias entre descriptores de dos nubes.
// Devuelve una lista de correspondencias entre dos nubes de descriptores.
template <typename DescriptorType>
pcl::CorrespondencesPtr findCorrespondences(
    const typename pcl::PointCloud<DescriptorType>::Ptr &desc_actual,
    const typename pcl::PointCloud<DescriptorType>::Ptr &desc_prev)
{
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    registration::CorrespondenceEstimation<DescriptorType, DescriptorType> est;
	est.setInputSource(desc_actual);
	est.setInputTarget(desc_prev);
	est.determineCorrespondences(*correspondences); // Aqui calculan las correspondencias entre los descriptores

    return correspondences;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// RANSAC (FILTRAR CORRESPONDENCIAS) ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Función que filtra correspondencias utilizando RANSAC.
// Obtiene la transformación a pesar de los "malos emparejamientos"
template<typename DescriptorT> void filterCorrespondencesRANSAC(
                        PointCloud<PointXYZRGB>::Ptr keypoints_actual,
                        PointCloud<PointXYZRGB>::Ptr keypoints_prev,
                        CorrespondencesPtr correspondences,
                        Eigen::Matrix4f &transformation)
{
    pcl::CorrespondencesPtr filtered(new pcl::Correspondences());
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> ransac;
    ransac.setInputSource(keypoints_actual);
    ransac.setInputTarget(keypoints_prev);
    ransac.setInputCorrespondences(correspondences); // Las correspondencias iniciales a refinar
    ransac.setInlierThreshold(0.2); // Si quiero hacerlo más estricto, disminuir este valor // 0.15 el mejor
    ransac.setMaximumIterations(5000); // Número máximo de iteraciones del algoritmo RANSAC
    ransac.setRefineModel(true); // Activa la opción de refinar el modelo de transformación final
    ransac.getCorrespondences(*filtered);// Aumentar el umbral para ser más permisivo

    // solo aplicable a Harris para evitar transformaciones bruscas
    if (filtered->size() < 10 && USE_HARRIS) {
        ROS_WARN("RANSAC falló: solo %lu inliers. Se descarta la transformación.", filtered->size());
        transformation = Eigen::Matrix4f::Identity(); // evitar basura acumulada
        return;
    }

    // Actualiza la transformación con la obtenida por RANSAC
    transformation = ransac.getBestTransformation();

    // Visualizar correspondencias filtradas
    cout << "CORRESPONDENCIAS TRAS APLICAR RANSAC: " << filtered->size() << endl;
    cout << "MATRIZ DE TRANSFORMACION ACTUALIZADA:\n" << transformation << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// ICP (TRANSFORMACIÓN) //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Refina la alineación entre dos nubes de puntos usando el algoritmo ICP (Iterative Closest Point)
// NO SIEMPRE LA USAMOS, ES UNA PRUEBA
Eigen::Matrix4f refineAlignmentICP(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target)
{
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.15);  // Aumenta este valor
    icp.setMaximumIterations(100);           // Más iteraciones
    icp.setTransformationEpsilon(1e-8);      // Mayor precisión
    
    pcl::PointCloud<pcl::PointXYZRGB> aligned;
    icp.align(aligned);
    
    std::cout << "ICP score: " << icp.getFitnessScore() << std::endl;
    
    return icp.getFinalTransformation();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// FUNCIONES DE PROCESAMIENTO ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Función para girar el robot para la toma de capturas
void rotateRobot(){
    if (steps_left <= 0) {
        return;
    }

    cout << "Steps left: " << steps_left << endl;
    geometry_msgs::Twist rotate_cmd;
    ros::Rate rate(10);
    
    // Girar el robot
    rotate_cmd.angular.z = angular_speed;
    rotate_cmd.linear.x = 0;
    cmd_vel_pub.publish(rotate_cmd);
    ros::Duration(STEP_DURATION).sleep();

    // Detener el giro
    rotate_cmd.angular.z = 0;
    cmd_vel_pub.publish(rotate_cmd);
    steps_left--;
}

// Función para obtener un timestamp en segundos
double getTimestamp() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

void process_pointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud) {

    // 4. CALCULO DE LAS NORMALES A PARTIR DE LA NUBE FILTRADA 
    pcl::PointCloud<pcl::Normal>::Ptr normals = compute_normals(cloud);

    // 5. DETECTAR KEYPOINTS
    PointIndicesConstPtr keypoint_index;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints = detect_keypoints(cloud, keypoint_index);

    // NEKO: std::string timestamp = std::to_string(ros::Time::now().toSec());
    std::string timestamp = std::to_string(getTimestamp());
    std::string keypoints_pcd_path = "./points/keypoints_" + timestamp + ".pcd";

    if (!keypoints->empty()) {
        pcl::io::savePCDFileASCII(keypoints_pcd_path, *keypoints);
    } else {
        ROS_WARN("No se detectaron keypoints, no se guardara el archivo.");
    }

    // 6. CALCULAR DESCRIPTORES
    if (USE_SHOT) {
        auto desc_SHOT = compute_shot_descriptor(cloud, normals, keypoint_index);

        // Verificar si hay una nube previa para encontrar correspondencias
        if (prev_desc_SHOT) {
            /*// 7. SACAR LAS CORRESPONDENCIAS
            auto correspondences = findCorrespondences<pcl::SHOT352>(desc_SHOT, prev_desc_SHOT);
            
            // 8. APLICAR RANSAC
            Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
            filterCorrespondencesRANSAC<SHOT352>(keypoints, prev_keypoints, correspondences, transformation);

            transformation_acumulada = transformation * transformation_acumulada;

            // Transformar los keypoints actuales con la transformación obtenida
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_acumulada);
            // Agregar la nube transformada al mapa global (pero no guardar aún)
            if (!transformed_cloud->empty()) {
                *global_map += *transformed_cloud;
            }*/
            // 1. Obtener correspondencias entre los descriptores
            auto correspondences = findCorrespondences<pcl::SHOT352>(desc_SHOT, prev_desc_SHOT);

            // 2. Estimar la transformación inicial con RANSAC
            Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
            filterCorrespondencesRANSAC<pcl::SHOT352>(keypoints, prev_keypoints, correspondences, transformation);

            // 3. Aplicar la transformación inicial a la nube actual
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation);

            if (!USE_HARRIS) {
                // 4. Refinar con ICP (usando la nube transformada y la anterior)
                Eigen::Matrix4f refined_transformation = refineAlignmentICP(transformed_cloud, prev_keypoints);
            
                // 5. Actualizar la transformación acumulada
                transformation_acumulada = refined_transformation * transformation * transformation_acumulada;
            } else {
                // Sin refinamiento: aplicar solo la transformación estimada por RANSAC
                transformation_acumulada = transformation * transformation_acumulada;
            }


            // 6. Aplicar la transformación acumulada final a la nube original
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_acumulada);

            // 7. Añadir al mapa global
            if (!transformed_cloud->empty()) {
                *global_map += *transformed_cloud;
            }
        }
        prev_desc_SHOT = desc_SHOT;
    } 
    else { // Usar FPFH
        auto desc_FPFH = compute_fpfh_descriptor(cloud, normals, keypoint_index);

        // Verificar si hay una nube previa para encontrar correspondencias
        if (prev_desc_FPFH) {
            // 1. Obtener correspondencias entre los descriptores
            auto correspondences = findCorrespondences<pcl::FPFHSignature33>(desc_FPFH, prev_desc_FPFH);

            // 2. Estimar la transformación inicial con RANSAC
            Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
            filterCorrespondencesRANSAC<pcl::FPFHSignature33>(keypoints, prev_keypoints, correspondences, transformation);

            // 3. Aplicar la transformación inicial a la nube actual
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation);

            if (!USE_HARRIS) {
                // 4. Refinar con ICP (usando la nube transformada y la anterior)
                Eigen::Matrix4f refined_transformation = refineAlignmentICP(transformed_cloud, prev_keypoints);
            
                // 5. Actualizar la transformación acumulada
                transformation_acumulada = refined_transformation * transformation * transformation_acumulada;
            } else {
                // Sin refinamiento: aplicar solo la transformación estimada por RANSAC
                transformation_acumulada = transformation * transformation_acumulada;
            }

            // 6. Aplicar la transformación acumulada final a la nube original
            pcl::transformPointCloud(*cloud, *transformed_cloud, transformation_acumulada);

            // 7. Añadir al mapa global
            if (!transformed_cloud->empty()) {
                *global_map += *transformed_cloud;
            }
        }
        prev_desc_FPFH = desc_FPFH;
    }
    prev_keypoints = keypoints;
}


void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &msg)
{
    if (steps_left <= 0) {
        return;
    }

    // 1. SACAR LOS PUNTOS (TODA LA NUBE)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*msg));
    ROS_INFO("Puntos capturados: %lu", cloud->size());
    
    // 2. QUITAR NANS:: Elimina errores en la medición o datos fuera del rango del sensor.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_no_nan = removeNaNPoints(cloud);

    // 3. APLICAR VOXELGRID SOBRE LOS PUNTOS FILTRADOS
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> vGrid;
    vGrid.setInputCloud(cloud_no_nan);
    vGrid.setLeafSize(0.01f, 0.01f, 0.01f);
    vGrid.filter(*cloud_filtered);

    ROS_INFO("Puntos tras VG: %lu", cloud_filtered->size());
    
    std::string timestamp = std::to_string(ros::Time::now().toSec());
    std::string filtered_pcd_path = "./points/points_" + timestamp + ".pcd";
    pcl::io::savePCDFileASCII(filtered_pcd_path, *cloud_filtered);

    process_pointcloud(cloud_filtered);

    boost::mutex::scoped_lock lock(cloud_mutex);
    visu_pc = cloud_filtered;
    lock.unlock();

    // Rotate the robot
    rotateRobot();

    // Si es el último paso, guardar el mapa 3D
    if (steps_left == 0) {
        pcl::PointCloud<PointXYZRGB>::Ptr nube_final(new PointCloud<PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> vGrid;
        vGrid.setInputCloud(global_map);    
        vGrid.setLeafSize(0.01f, 0.01f, 0.01f);
        vGrid.filter(*nube_final);
        // Guardar el mapa 3D solo una vez al final
        if (!global_map->empty()) {
            pcl::io::savePCDFileASCII("./points/global_map.pcd", *nube_final);
            ROS_INFO("Mapa 3D final guardado con %lu puntos", global_map->size());
        } else {
            ROS_WARN("El mapa global está vacío. No se guardará el archivo.");
        }
    }
}

int main(int argc, char **argv)
{
    if (!USE_TAKEN_POINTS) {
        ros::init(argc, argv, "sub_pcl");
        ros::NodeHandle nh;
        boost::thread viewer_thread(simpleVis);
        ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB>>("/camera/depth/points", 1, callback);
        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        system("rosservice call /gazebo/reset_simulation");
    
        // Este bucle mantendrá el nodo activo
        ros::Rate loop_rate(10); // 10 Hz
        while (ros::ok() && steps_left > 0) {
            ros::spinOnce();
            loop_rate.sleep();
        }
    } else {
        // Leer todos los archivos de puntos en la carpeta
        vector<string> fileList;
        boost::filesystem::path dir("./points");
        if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
            for (auto& entry : boost::filesystem::directory_iterator(dir)) {
                string filename = entry.path().filename().string();
                if (filename.find("points_") == 0 && entry.path().extension() == ".pcd") {
                    fileList.push_back(entry.path().string());
                }
            }
        } else {
            cerr << "La carpeta 'points' no existe o no es un directorio." << endl;
            return 0;
        }

        // Ordenar los archivos de manera numérica
        sort(fileList.begin(), fileList.end(), [](const string &a, const string &b) {
            int numA = stoi(a.substr(a.find_last_of('_') + 1, a.find_last_of('.') - a.find_last_of('_') - 1));
            int numB = stoi(b.substr(b.find_last_of('_') + 1, b.find_last_of('.') - b.find_last_of('_') - 1));
            return numA < numB;
        });
        if (fileList.empty()) {
            cerr << "No se encontraron archivos de nubes de puntos." << endl;
            return 0;
        }

        // Iterar sobre cada archivo de nube de puntos
        for (const auto& file : fileList) {
            cout << "Procesando archivo " << file << endl;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(file, *cloud) == -1) {
                cerr << "Error al cargar el archivo " << file << endl;
                return 0;
            }
            process_pointcloud(cloud);
        }

        pcl::PointCloud<PointXYZRGB>::Ptr nube_final(new PointCloud<PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> vGrid;
        vGrid.setInputCloud(global_map);    
        vGrid.setLeafSize(0.01f, 0.01f, 0.01f);
        vGrid.filter(*nube_final);
        // Guardar el mapa 3D solo una vez al final
        if (!global_map->empty()) {
            pcl::io::savePCDFileASCII("./points/global_map.pcd", *nube_final);
            ROS_INFO("Mapa 3D final guardado con %lu puntos", global_map->size());
        } else {
            ROS_WARN("El mapa global está vacío. No se guardará el archivo.");
        }
    }

    return 0;
}