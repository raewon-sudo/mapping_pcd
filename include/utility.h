#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <geographic_msgs/GeoPointStamped.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <lanelet2_core/primitives/Point.h>
#include <lanelet2_core/primitives/GPSPoint.h>
#include <lanelet2_extension/projection/mgrs_projector.h>

#include <boost/filesystem.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef std::numeric_limits< double > dbl;

typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZINormal PointType2;
typedef pcl::PointCloud<PointType2> PointCloudXYZI;

enum class SensorType { VELODYNE, OUSTER, PANDAR, LIVOX_HORIZON };
enum class LoopType { ICP, GICP, NDT };
enum class projectmapType { UTM, MGRS };
enum class modeSC { normal, intensity };

class ParamServer
{
public:

    ros::NodeHandle nh;

    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsOdom_Topic;
    string gpsTopic;

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;
    projectmapType map_project;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float gpsCovThreshold_z;
    float poseCovThreshold;
    bool useNavsat;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;
    string loadmapDirectory;
    string file_BIN;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;
    bool useCloudRing;
    float ang_res_x;
    float ang_res_y;
    float ang_bottom;
    int groundScanInd;

    // IMU
    float imuRate;  //fix ngay 12/5/2021
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;
    Eigen::Vector3d gyr_prev;   //fix ngay 12/5/2021

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize;
    float kSCLeafSize;
    float kDownsampleVoxelSize;

    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    //Scancontext loop
    bool   loopClosureSCEnableFlag;
    modeSC mode_SC;
    double pc_max_radius;
    double pc_dist_thresh;
    float historyKeyframeFitnessScore_SC;

    // Loop closure
    LoopType loop_algoth;
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore_RS;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer()
    {
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>("mapping_pcd/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("mapping_pcd/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("mapping_pcd/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("mapping_pcd/gpsOdom_Topic", gpsOdom_Topic, "odometry/gps");
        nh.param<std::string>("mapping_pcd/gpsTopic", gpsTopic, "sensing/gnss/ublox/nav_sat_fix");

        nh.param<std::string>("mapping_pcd/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("mapping_pcd/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("mapping_pcd/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("mapping_pcd/mapFrame", mapFrame, "map");

        std::string map_projectStr;
        nh.param<std::string>("mapping_pcd/map_project", map_projectStr, "");
        if (map_projectStr == "utm")
        {
            map_project = projectmapType::UTM;
        }
        else if (map_projectStr == "mgrs")
        {
            map_project = projectmapType::MGRS;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid project map type (must be either 'utm' or 'mgrs' ): " << map_projectStr);
            ros::shutdown();
        }
        ROS_INFO_COND(int(map_project) == 0, "\033[3;36m----> Project su dung: UTM. \033[0m");
        ROS_INFO_COND(int(map_project) == 1, "\033[3;36m----> Project su dung: MGRS. \033[0m");

        nh.param<bool>("mapping_pcd/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("mapping_pcd/useGpsElevation", useGpsElevation, false);
        nh.param<float>("mapping_pcd/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("mapping_pcd/gpsCovThreshold_z", gpsCovThreshold_z, 2.0);
        nh.param<float>("mapping_pcd/poseCovThreshold", poseCovThreshold, 25.0);
        nh.param<bool>("mapping_pcd/useNavsat", useNavsat, true);

        nh.param<bool>("mapping_pcd/savePCD", savePCD, false);
        nh.param<std::string>("mapping_pcd/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");
        nh.param<std::string>("mapping_pcd/loadmapDirectory", loadmapDirectory, "/Autoware_IV_file/map_folder/");
        nh.param<std::string>("mapping_pcd/file_BIN", file_BIN, "BIN_files");

        std::string sensorStr;
        nh.param<std::string>("mapping_pcd/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "pandar")
        {
            sensor = SensorType::PANDAR;
        }
        else if (sensorStr == "livox_horizon")
        {
            sensor = SensorType::LIVOX_HORIZON;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'pandar' or 'livox_horizon'): " << sensorStr);
            ros::shutdown();
        }
        ROS_INFO_COND(int(sensor) == 0, "\033[2;39m----> Lidar su dung: Velodyne. \033[0m");
        ROS_INFO_COND(int(sensor) == 1, "\033[2;39m----> Lidar su dung: Ouster. \033[0m");
        ROS_INFO_COND(int(sensor) == 2, "\033[2;39m----> Lidar su dung: Hesai. \033[0m");
        ROS_INFO_COND(int(sensor) == 3, "\033[2;39m----> Lidar su dung: Livox_horizon. \033[0m");

        nh.param<int>("mapping_pcd/N_SCAN", N_SCAN, 16);
        nh.param<int>("mapping_pcd/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("mapping_pcd/downsampleRate", downsampleRate, 1);
        nh.param<float>("mapping_pcd/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("mapping_pcd/lidarMaxRange", lidarMaxRange, 1000.0);
        nh.param<bool>("mapping_pcd/useCloudRing", useCloudRing, true);
        nh.param<float>("mapping_pcd/ang_res_x", ang_res_x, 360/float(Horizon_SCAN));
        nh.param<float>("mapping_pcd/ang_res_y", ang_res_y, 2.0);
        nh.param<float>("mapping_pcd/ang_bottom", ang_bottom, 15.0+0.1);
        nh.param<int>("mapping_pcd/groundScanInd", groundScanInd, 8);

        nh.param<float>("lio_sam/imuRate", imuRate, 200.0);
        nh.param<float>("mapping_pcd/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("mapping_pcd/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("mapping_pcd/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("mapping_pcd/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("mapping_pcd/imuGravity", imuGravity, 9.80511);
        nh.param<float>("mapping_pcd/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<vector<double>>("mapping_pcd/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("mapping_pcd/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("mapping_pcd/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        nh.param<float>("mapping_pcd/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("mapping_pcd/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("mapping_pcd/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("mapping_pcd/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("mapping_pcd/odometrySurfLeafSize", odometrySurfLeafSize, 0.4);
        nh.param<float>("mapping_pcd/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("mapping_pcd/mappingSurfLeafSize", mappingSurfLeafSize, 0.4);
        nh.param<float>("mapping_pcd/kSCLeafSize", kSCLeafSize, 0.2);
        nh.param<float>("mapping_pcd/kDownsampleVoxelSize", kDownsampleVoxelSize, 0.05);

        nh.param<float>("mapping_pcd/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("mapping_pcd/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("mapping_pcd/numberOfCores", numberOfCores, 2);
        nh.param<double>("mapping_pcd/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>("mapping_pcd/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("mapping_pcd/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("mapping_pcd/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("mapping_pcd/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        std::string loop_algothStr;
        nh.param<std::string>("mapping_pcd/loop_algoth", loop_algothStr, "");
        if (loop_algothStr == "icp")
        {
            loop_algoth = LoopType::ICP;
        }
        else if (loop_algothStr == "gicp")
        {
            loop_algoth = LoopType::GICP;
        }
        else if (loop_algothStr == "ndt")
        {
            loop_algoth = LoopType::NDT;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Khong chon thuat toan Loop (Can dien thuat toan 'icp' hoac 'gicp' hoac 'ndt'): " << loop_algothStr);
            ros::shutdown();
        }

        ROS_INFO_COND(int(loop_algoth) == 0, "\033[0;34m----> Thuat toan loop su dung: ICP. \033[0m");
        ROS_INFO_COND(int(loop_algoth) == 1, "\033[0;34m----> Thuat toan loop su dung: GICP. \033[0m");
        ROS_INFO_COND(int(loop_algoth) == 2, "\033[0;34m----> Thuat toan loop su dung: NDT. \033[0m");

        std::string mode_SCStr;
        nh.param<std::string>("mapping_pcd/mode_SC", mode_SCStr, "");
        if (mode_SCStr == "normal")
        {
            mode_SC = modeSC::normal;
        }
        else if (mode_SCStr == "intensity")
        {
            mode_SC = modeSC::intensity;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Khong chon thuat toan scancontext: " << mode_SCStr);
            ros::shutdown();
        }

        nh.param<double>("mapping_pcd/pc_dist_thresh", pc_dist_thresh, 0.6);
        nh.param<double>("mapping_pcd/pc_max_radius", pc_max_radius, 80.0);
        nh.param<float>("mapping_pcd/historyKeyframeFitnessScore_SC", historyKeyframeFitnessScore_SC, 0.3);

        nh.param<bool>("mapping_pcd/loopClosureSCEnableFlag", loopClosureSCEnableFlag, false);

        if(loopClosureSCEnableFlag == true)
        {
            ROS_INFO("\033[0;34m----> Co su dung Scancontext \033[0m");
        }
        else
        {
            ROS_INFO("\033[0;34m----> Khong su dung Scancontext \033[0m");
        }

        nh.param<bool>("mapping_pcd/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("mapping_pcd/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("mapping_pcd/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("mapping_pcd/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("mapping_pcd/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("mapping_pcd/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("mapping_pcd/historyKeyframeFitnessScore_RS", historyKeyframeFitnessScore_RS, 0.3);

        nh.param<float>("mapping_pcd/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("mapping_pcd/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("mapping_pcd/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // rotate acceleration
        //fix ngay 12/5/2021
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();
        // // rotate acceleration
        // Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        // acc = extRot * acc;
        // //acc = acc gan nhat + delta_gocquay_IMU/T * Matran_tinh_tien_IMU_lidar + gocquay_IMU*(gocquay_IMU * Matran_tinh_tien_IMU_lidar)
        // acc = acc + ((gyr - gyr_prev) * imuRate).cross(-extTrans) + gyr.cross(gyr.cross(-extTrans));

        // imu_out.linear_acceleration.x = acc.x();
        // imu_out.linear_acceleration.y =  acc.y();
        // imu_out.linear_acceleration.z = acc.z();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        // gyr_prev = gyr;   //fix ngay 12/5/2021
        return imu_out;
    }
};

sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

sensor_msgs::PointCloud2 publishCloud2(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T> //Hàm chuyển giá trị IMU ra dữ liệu van tốc góc
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T> //Hàm chuyển giá trị IMU ra dữ liệu gia toc tinh tiến
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T> //Hàm chuyển giá trị IMU ra dữ liệu RPY
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");

    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

void saveBINfile(std::string filename, pcl::PointCloud<PointType>::Ptr key_frame)
{
            // ref from gtsam's original code "dataset.cpp"
            // std::fstream stream(filename.c_str(), fstream::out);
    std::ofstream file(filename.c_str(),ios::out|ios::binary|ios::app);
    if (file.is_open())
    {
        for (int i=0;i<key_frame->points.size();i++)
        {
            file.write((char*)&key_frame-> points[i].x,sizeof(float));
            file.write((char*)&key_frame-> points[i].y,sizeof(float));
            file.write((char*)&key_frame-> points[i].z,sizeof(float));
            file.write((char*)&key_frame-> points[i].intensity,sizeof(float));
            // stream << arr[4] << std::endl;
        }
        file.close();
    }
}

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;     //Điền giá trị 0 vào các vị trí trông với setfill đi kèm với setw
  return out.str();
}

void writing_file_save(std::string filename)
{
    if(boost::filesystem::is_directory(filename) == 0)          //Check ton tai cua thu muc
    {
        boost::filesystem::create_directory(filename);
    }
    else if (boost::filesystem::is_directory(filename) == 1)
    {
        boost::filesystem::remove_all(filename);
        boost::filesystem::create_directory(filename);
    }
}

#endif
