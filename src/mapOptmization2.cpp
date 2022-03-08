#include "utility.h"
#include "mapping_pcd/cloud_info.h"   //Them message custom moi
#include "mapping_pcd/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <geodesy/utm.h>
#include <geodesy/wgs84.h>

#include "Scancontext.h"
#include <pcl/octree/octree_search.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>      //Thêm ngày 4/12/2021

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)   Khai bao diem
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)    Khai bao van toc
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)     Khai bao diem dich chuyen
using gtsam::symbol_shorthand::G; // GPS pose    Khai bao vi tri GPS

//-----------------------------------------------------------------------------------------------------------------
void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3>*>(&key_value.value);
        if (!p) continue;

        const gtsam::Pose3& pose = p->value();

        gtsam::Point3 t = pose.translation();
        gtsam::Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

//Thêm ngày 4/12/2021 -- Test Octree
void octreeDownsampling(const pcl::PointCloud<PointType>::Ptr& _src, pcl::PointCloud<PointType>::Ptr& _to_save, float kDownsampleVoxelSize)
{
    pcl::octree::OctreePointCloudVoxelCentroid<PointType> octree( kDownsampleVoxelSize );
    octree.setInputCloud(_src);
    octree.defineBoundingBox();
    octree.addPointsFromInputCloud();
    pcl::octree::OctreePointCloudVoxelCentroid<PointType>::AlignedPointTVector centroids;
    octree.getVoxelCentroids(centroids);

    // init current map with the downsampled full cloud
    _to_save->points.assign(centroids.begin(), centroids.end());
    _to_save->width = 1;
    _to_save->height = _to_save->points.size(); // make sure again the format of the downsampled point cloud
    ROS_INFO_STREAM("\033[1;32m Downsampled pointcloud have: " << _to_save->points.size() << " points.\033[0m");
    cout << endl;
} // octreeDownsampling

//----------------------------------------------------------------------------------------------------------

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

//-------------------------------------------------------------------------
enum class SCInputType {
    SINGLE_SCAN_FULL,
    SINGLE_SCAN_FEAT,
    MULTI_SCAN_FEAT
};
//--------------------------------------------------------------------------

class mapOptimization : public ParamServer
{

public:

    // gtsam
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubLoopScanLocal;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher image_sc_pub;
    ros::Publisher pub_gps_truth;

    ros::Subscriber subCloud;
    ros::Subscriber subOdom_GPS;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    std::deque<geographic_msgs::GeoPointStamped> gpsTopicQueue;
    boost::optional<Eigen::Vector3d> zero_projector;
    mapping_pcd::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D_utm;
    //-------------------------------------------------------------------
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D;
    //--------------------------------------------------------------------

    //--------------------------------------------------------------------------------
    pcl::PointCloud<PointType>::Ptr laserCloudRaw; // giseop
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS; // giseop
    double laserCloudRawTime;
    //------------------------------------------------------------------------------

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
    pcl::octree::OctreePointCloudSearch<PointType>::Ptr octreeSurroundingKeyPoses;
    pcl::octree::OctreePointCloudSearch<PointType>::Ptr octreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSC;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    //Check cờ Loop
    bool aLoopIsClosed = false;
    bool aGPSIsClosed = false;

    // map<int, int> loopIndexContainer; // from new to old
    std::multimap<int, int> loopIndexContainer;
    std::vector<std::pair<int, int>> loopIndexQueue;
    std::vector<gtsam::Pose3> loopPoseQueue;
    // vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    std::vector<gtsam::SharedNoiseModel> loopNoiseQueue;
    std::deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

//---------------------------------------------------------------------------------
    // // loop detector
    SCManager scManager;

    // data saver
    std::fstream pgSaveStream;     // pg: pose-graph
    std::fstream pgTimeSaveStream; // pg: pose-graph
    std::vector<std::string> edges_str;
    std::vector<std::string> vertices_str;
    // std::fstream pgVertexSaveStream;
    // std::fstream pgEdgeSaveStream;

    std::string saveProcessDirectory;
    std::string saveSCDDirectory;
    std::string saveNodePCDDirectory;
    std::string saveNodeBINDirectory;
//--------------------------------------------------------------------------------

public:
    mapOptimization()
    {
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new gtsam::ISAM2(parameters);

        //Tham số Scancontext
        scManager.setSCdistThres(pc_dist_thresh);
        scManager.setMaximumRadius(pc_max_radius);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("mapping_pcd/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("mapping_pcd/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("mapping_pcd/mapping/path", 1);

        subCloud    = nh.subscribe<mapping_pcd::cloud_info>("mapping_pcd/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom_GPS = nh.subscribe<nav_msgs::Odometry> (gpsOdom_Topic, 200, &mapOptimization::gpsOdomHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS      = nh.subscribe<sensor_msgs::NavSatFix> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoop     = nh.subscribe<std_msgs::Float64MultiArray>("mapping_pcd/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        srvSaveMap  = nh.advertiseService("mapping_pcd/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/icp_loop_closure_history_cloud", 1);
        pubLoopScanLocal      = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/loop_scan_local", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/mapping_pcd/mapping/loop_closure_constraints", 1);
        //------------------------------------------------------------------------
        image_sc_pub          = nh.advertise<sensor_msgs::Image>("/image_SC", 1);
        pub_gps_truth         = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/gps_truth", 1);
        //-------------------------------------------------------------------------

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/mapping/cloud_registered_raw", 1);

        //Downsize voxel grid filter
        downSizeFilterSC.setLeafSize(kSCLeafSize, kSCLeafSize, kSCLeafSize);
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();

        //-----------------------------------------------------------------------------------
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

        // giseop
        // create directory and remove old files;
        saveProcessDirectory = std::getenv("HOME") + savePCDDirectory; // rather use global path
        int unused = system((std::string("exec rm -r ") + saveProcessDirectory).c_str());
        unused = system((std::string("mkdir ") + saveProcessDirectory).c_str());

        saveSCDDirectory = saveProcessDirectory + "SCDs/"; // SCD: scan context descriptor
        unused = system((std::string("exec rm -r ") + saveSCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveSCDDirectory).c_str());

        saveNodePCDDirectory = saveProcessDirectory + "Scans/";
        unused = system((std::string("exec rm -r ") + saveNodePCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveNodePCDDirectory).c_str());

        saveNodeBINDirectory = saveProcessDirectory + "BIN_files/";
        unused = system((std::string("exec rm -r ") + saveNodeBINDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveNodeBINDirectory).c_str());

        pgSaveStream = std::fstream(saveProcessDirectory + "singlesession_posegraph.g2o", std::fstream::out);
        pgTimeSaveStream = std::fstream(saveProcessDirectory + "times.txt", std::fstream::out);
        pgTimeSaveStream.precision(dbl::max_digits10);
        // pgVertexSaveStream = std::fstream(savePCDDirectory + "singlesession_vertex.g2o", std::fstream::out);
        // pgEdgeSaveStream = std::fstream(savePCDDirectory + "singlesession_edge.g2o", std::fstream::out);
        //-----------------------------------------------------------------------------------------------------------------------------------
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        cloudKeyPoses6D_utm.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudRaw.reset(new pcl::PointCloud<PointType>());   // giseop
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>()); // giseop

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());   // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());     // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void writeVertex(const int _node_idx, const gtsam::Pose3 &_initPose)
    {
        gtsam::Point3 t = _initPose.translation();
        gtsam::Rot3 R = _initPose.rotation();

        std::string curVertexInfo{
            "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " " + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z()) + " " + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w())};

        // pgVertexSaveStream << curVertexInfo << std::endl;
        vertices_str.emplace_back(curVertexInfo);
    }

    void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3 &_relPose)
    {
        gtsam::Point3 t = _relPose.translation();
        gtsam::Rot3 R = _relPose.rotation();

        std::string curEdgeInfo{
            "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " " + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z()) + " " + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w())};

        // pgEdgeSaveStream << curEdgeInfo << std::endl;
        edges_str.emplace_back(curEdgeInfo);
    }

    // void writeEdgeStr(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, const gtsam::SharedNoiseModel _noise)
    // {
    //     gtsam::Point3 t = _relPose.translation();
    //     gtsam::Rot3 R = _relPose.rotation();

    //     std::string curEdgeSaveStream;
    //     curEdgeSaveStream << "EDGE_SE3:QUAT " << _node_idx_pair.first << " " << _node_idx_pair.second << " "
    //         << t.x() << " "  << t.y() << " " << t.z()  << " "
    //         << R.toQuaternion().x() << " " << R.toQuaternion().y() << " " << R.toQuaternion().z()  << " " << R.toQuaternion().w() << std::endl;

    //     edges_str.emplace_back(curEdgeSaveStream);
    // }

    void laserCloudInfoHandler(const mapping_pcd::cloud_infoConstPtr &msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);  //Lấy ra đối tượng Corner và Surf sau quá trinh feature extraction
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        //------------------------------------------------------------------------------------
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw);
        laserCloudRawTime = cloudInfo.header.stamp.toSec();     // save node time
        //-----------------------------------------------------------------------------------------

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsOdomHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        //---------------------------------------------------------------- 21/10/2021
        //Xuất ra bản tin GPS dạng geometry_msgs/Pose
        geometry_msgs::PoseWithCovarianceStamped gps_truth;

        gps_truth.pose.pose.position.x = gpsMsg->pose.pose.position.x;
        gps_truth.pose.pose.position.y = gpsMsg->pose.pose.position.y;
        gps_truth.pose.pose.position.z = gpsMsg->pose.pose.position.z;

        gps_truth.pose.covariance[0] = gpsMsg->pose.covariance[0];
        gps_truth.pose.covariance[7] = gpsMsg->pose.covariance[7];
        gps_truth.pose.covariance[14] = gpsMsg->pose.covariance[14];

        gps_truth.header.stamp = timeLaserInfoStamp;
        gps_truth.header.frame_id = odometryFrame;

        pub_gps_truth.publish(gps_truth);
        //--------------------------------------------------------------------

        gpsQueue.push_back(*gpsMsg);    //Day du lieu tu GPS Topic -> gpsQueue TG
        // cout << "Dang xu ly du lieu GPS" << endl;
    }

    void gpsHandler(const sensor_msgs::NavSatFixConstPtr& gpsTopicMsgIn)
    {
        //--------------Test 1/6/2021 cho toa do dia cau
        geographic_msgs::GeoPointStampedPtr gps_msg(new geographic_msgs::GeoPointStamped());
        gps_msg->header = gpsTopicMsgIn->header;
        gps_msg->position.latitude = gpsTopicMsgIn->latitude;
        gps_msg->position.longitude = gpsTopicMsgIn->longitude;
        gps_msg->position.altitude = gpsTopicMsgIn->altitude;

        gpsTopicQueue.push_back(*gps_msg);
    }

    void pointAssociateToMap(PointType const *const pi, PointType *const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];
        return thisPose6D;
    }

//Them code save map rieng
    bool saveMapService(mapping_pcd::save_mapRequest& req, mapping_pcd::save_mapResponse& res)
    {
        string saveMapDirectory;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        if(req.destination.empty())
        {
            saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
        }
        else
        {
            saveMapDirectory = std::getenv("HOME") + req.destination;
            // create directory and remove old files;
            int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
            unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
        }
        cout << "Save destination: " << saveMapDirectory << endl;

        // save pose graph (runs when programe is closing)
        cout << "****************************************************" << endl;
        cout << "Saving the posegraph ..." << endl; // giseop

        for (auto &_line : vertices_str)
            pgSaveStream << _line << std::endl;
        for (auto &_line : edges_str)
            pgSaveStream << _line << std::endl;

        pgSaveStream.close();
        // pgVertexSaveStream.close();
        // pgEdgeSaveStream.close();

        const std::string kitti_format_pg_filename{saveMapDirectory + "optimized_poses.txt"};
        saveOptimizedVerticesKITTIformat(isamCurrentEstimate, kitti_format_pg_filename);

        // save key frame transformations ->
        pcl::io::savePCDFileBinary(saveMapDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileBinary(saveMapDirectory + "transformations.pcd", *cloudKeyPoses6D);

        // extract global point cloud map
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr octree_filter_globalmap(new pcl::PointCloud<PointType>());

        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
            //fake toa do theo utm
            *cloudKeyPoses6D_utm = *cloudKeyPoses6D;

            cloudKeyPoses6D_utm->points[i].x = cloudKeyPoses6D->points[i].x + zero_projector->x();
            cloudKeyPoses6D_utm->points[i].y = cloudKeyPoses6D->points[i].y + zero_projector->y();
            cloudKeyPoses6D_utm->points[i].z = cloudKeyPoses6D->points[i].z + zero_projector->z();

            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D_utm->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D_utm->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D_utm->size() << " ...";
        }

        //Danh cho tọa độ địa tâm
        // for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
        //     //fake toa do theo utm
        //     *cloudKeyPoses6D_utm = *cloudKeyPoses6D;

        //     cloudKeyPoses6D_utm->points[i].x = cloudKeyPoses6D->points[i].x + zero_projector->x();
        //     cloudKeyPoses6D_utm->points[i].y = cloudKeyPoses6D->points[i].y + zero_projector->y();
        //     cloudKeyPoses6D_utm->points[i].z = cloudKeyPoses6D->points[i].z + zero_projector->z();

        //     *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D_utm->points[i]);
        //     *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D_utm->points[i]);
        //     cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D_utm->size() << " ...";
        // }

        if(zero_projector)
        {
            std::cout << "Ghi projector" << std::endl;
            std::ofstream ofs(saveMapDirectory + "map_projector.project");
            if(map_project == projectmapType::MGRS)
            {
                ofs << "MGRS" << std::endl;
            }else if(map_project == projectmapType::UTM)
            {
                ofs << "UTM" << std::endl;
            }

            ofs << boost::format("%.6f") % zero_projector->x() << std::endl;
            ofs << boost::format("%.6f") % zero_projector->y() << std::endl;
            ofs << boost::format("%.6f") % zero_projector->z() << std::endl;
        }

        if(req.resolution != 0)
        {
            cout << "\n\nSave resolution: " << req.resolution << endl;

            //Khoi tao de giam kich thuoc voxel corner cloud
            downSizeFilterCorner.setInputCloud(globalCornerCloud);
            //Giam kich thuoc point
            downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterCorner.filter(*globalCornerCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "CornerMap.pcd", *globalCornerCloudDS);
            //Khoi tao de giam kich thuoc voxel surf cloud
            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            //Giam kich thuoc point
            downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "SurfMap.pcd", *globalSurfCloudDS);
        }
        else
        {
            // save corner cloud
            pcl::io::savePCDFileBinary(saveMapDirectory + "CornerMap.pcd", *globalCornerCloud);
            // save surf cloud
            pcl::io::savePCDFileBinary(saveMapDirectory + "SurfMap.pcd", *globalSurfCloud);
        }

        // Ghep point tu Surf va Corner vao Global Map de save global point cloud map full
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;

        octreeDownsampling(globalMapCloud, octree_filter_globalmap, kDownsampleVoxelSize);

        // int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "GlobalMap.pcd", *globalMapCloud);
        int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "GlobalMap.pcd", *octree_filter_globalmap);     //Thêm ngày 4/12/2021 cho Octree map
        //res -> Phan hoi tu service
        res.success = ret == 0;     //Da save thanh cong

        cout << "****************************************************" << endl;
        cout << "Luu lai ban do thanh cong\n" << endl;

        return true;
    }

    //Hien thi Pointcloud len RVIZ
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        mapping_pcd::save_mapRequest  req;
        mapping_pcd::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }
// publish hien thi map duoc tao Map(global)
    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;      //pointSearchIndGlobalMap[0] là GlobalMap_idx
        }

        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        //---------- downsample visualized points ---------------
        // Tao doi tuong can Voxel filter
        // pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        // downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        // downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        // downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

        //Octree Centroid filter
        octreeDownsampling(globalMapKeyFrames, globalMapKeyFramesDS, kDownsampleVoxelSize);

        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performRSLoopClosure();

            if(loopClosureSCEnableFlag == true)
            {
                if(mode_SC == modeSC::normal)
                {
                    performSCLoopClosure();
                    // ROS_INFO("\033[0;34m----> Thuat toan Scancontext su dung: Normal. \033[0m");

                }else if (mode_SC == modeSC::intensity)
                {
                    performSCLoopClosure_intensity();
                    // ROS_INFO("\033[0;34m----> Thuat toan Scancontext su dung: Intensity. \033[0m");
                }
            }

            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performRSLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        copy_cloudKeyPoses2D->clear();
        *copy_cloudKeyPoses2D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)          //Kiểm tra xem đã có Loop nào được tìm thấy chưa
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        std::cout << "RS loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());

        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);  //Lấy luôn điểm Key cuối đang check loop làm tâm point hiện tại
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);  //Lấy Key từ các điểm lân cận thỏa mãn số lượng xác định làm tâm point map cũ
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;

        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);

        // Cai dat thuat toan bat diem tuong dong ICP/GICP/NDT để tìm các điểm Loop
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        pcl::PointCloud<PointType>::Ptr icp_unused_result_2(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr icp_unused_result(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr icp_closed_cloud(new pcl::PointCloud<PointType>());

        static pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        pcl::PointCloud<PointType>::Ptr gicp_unused_result(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr gicp_closed_cloud(new pcl::PointCloud<PointType>());

        static pclomp::NormalDistributionsTransform<PointType, PointType> ndt;
        pcl::PointCloud<PointType>::Ptr ndt_unused_result(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr ndt_closed_cloud(new pcl::PointCloud<PointType>());
        float transform_probability_ndt;

        // Chi tiet thuat toan lua chon
        if (loop_algoth == LoopType::ICP)
        {
            //Thong so cai dat
            icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // ndt.setMaximumIterations(100);
            // ndt.setTransformationEpsilon(0.01);
            // ndt.setStepSize (0.1);
            // ndt.setResolution(1.0);
            // ndt.setRANSACIterations(0);

            // // Align clouds
            // ndt.setInputSource(cureKeyframeCloud);
            // ndt.setInputTarget(prevKeyframeCloud);
            // ndt.align(*icp_unused_result_2);

            // Align clouds
            icp.setInputSource(cureKeyframeCloud);
            icp.setInputTarget(prevKeyframeCloud);
            // icp.align(*icp_unused_result, ndt.getFinalTransformation());
            icp.align(*icp_unused_result);

            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::transformPointCloud(*cureKeyframeCloud, *icp_closed_cloud, icp.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, icp_closed_cloud, timeLaserInfoStamp, odometryFrame);
            }
        }

        else if (loop_algoth == LoopType::GICP)
        {
            //Thong so cai dat
            gicp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
            gicp.setMaximumIterations(100);
            gicp.setTransformationEpsilon(1e-8);
            gicp.setEuclideanFitnessEpsilon(1e-6);
            gicp.setRANSACIterations(0);
            gicp.setCorrespondenceRandomness(20);

            // Align clouds
            gicp.setInputSource(cureKeyframeCloud);
            gicp.setInputTarget(prevKeyframeCloud);
            gicp.align(*gicp_unused_result);

            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::transformPointCloud(*cureKeyframeCloud, *gicp_closed_cloud, gicp.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, gicp_closed_cloud, timeLaserInfoStamp, odometryFrame);
            }
        }

        else if (loop_algoth == LoopType::NDT)
        {
            //Thong so cai dat
            ndt.setMaximumIterations(100);
            ndt.setTransformationEpsilon(1e-6);
            ndt.setResolution(2.0);
            ndt.setRANSACIterations(0);
            ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);

            // Align clouds
            ndt.setInputSource(cureKeyframeCloud);
            ndt.setInputTarget(prevKeyframeCloud);
            ndt.align(*ndt_unused_result);

            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::transformPointCloud(*cureKeyframeCloud, *ndt_closed_cloud, ndt.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, ndt_closed_cloud, timeLaserInfoStamp, odometryFrame);
            }
            transform_probability_ndt = ndt.getTransformationProbability();
            std::cout << "----------------------------------------------" << std::endl;
            std::cout <<  "Chi so matching map: " << transform_probability_ndt << std::endl;

            size_t kichthuoc_matching = ndt.getFinalNumIteration();
            std::cout <<  "Kich thuoc matching: " << kichthuoc_matching << std::endl;
        }

        else
        {
            ROS_ERROR_STREAM("Khong co loai thuat toan su dung: " << int(loop_algoth));
            ros::shutdown();
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        if (loop_algoth == LoopType::ICP)
        {
            correctionLidarFrame = icp.getFinalTransformation();
        }
        else if (loop_algoth == LoopType::GICP)
        {
            correctionLidarFrame = gicp.getFinalTransformation();
        }
        else if (loop_algoth == LoopType::NDT)
        {
            correctionLidarFrame = ndt.getFinalTransformation();
        }

        // world frame: W
        // pose current: C
        // Vi tri cu cua loop frame: H

        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// H2C * W2H
        // tCorrect: Vi tri cua keyframe hien tai trong W -> pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore;
        if (loop_algoth == LoopType::ICP)
        {
            noiseScore = icp.getFitnessScore();

            if (icp.hasConverged() == false || noiseScore > historyKeyframeFitnessScore_RS)
            {
                std::cout << "ICP fitness test failed (" << noiseScore << " > " << historyKeyframeFitnessScore_RS << "). Reject this RS loop." << std::endl;
                return;
            }
            else
            {
                std::cout << "ICP fitness test passed (" << noiseScore << " < " << historyKeyframeFitnessScore_RS << "). Add this RS loop." << std::endl;
            }
        }
        else if (loop_algoth == LoopType::GICP)
        {
            noiseScore = gicp.getFitnessScore();

            if (gicp.hasConverged() == false || noiseScore > historyKeyframeFitnessScore_RS)
            {
                std::cout << "GICP fitness test failed (" << noiseScore << " > " << historyKeyframeFitnessScore_RS << "). Reject this RS loop." << std::endl;
                return;
            }
            else
            {
                std::cout << "GICP fitness test passed (" << noiseScore << " < " << historyKeyframeFitnessScore_RS << "). Add this RS loop." << std::endl;
            }
        }
        else if (loop_algoth == LoopType::NDT)
        {
            noiseScore = ndt.getFitnessScore();

            if (ndt.hasConverged() == false || noiseScore > historyKeyframeFitnessScore_RS || transform_probability_ndt < 15.0)
            {
                std::cout << "NDT fitness test failed (" << noiseScore << " > " << historyKeyframeFitnessScore_RS << "). Reject this RS loop." << std::endl;
                return;
            }
            else
            {
                std::cout << "NDT fitness test passed (" << noiseScore << " < " << historyKeyframeFitnessScore_RS << "). Add this RS loop." << std::endl;
            }
        }

        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
    }

    void performSCLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        int loopKeyCur, loopKeyPre;

        // find keys
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
        loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        loopKeyPre = detectResult.first;
        float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
        if (loopKeyPre == -1 /* No loop found */)
            return;

        std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, loopKeyPre);
        // loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, loopKeyPre);

        // loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        // loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

        int base_key = 0;
        loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, base_key);
        loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key);

        // loop verification
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        if (pubLoopScanLocal.getNumSubscribers() != 0)
            publishCloud(&pubLoopScanLocal, cureKeyframeCloud, timeLaserInfoStamp, odometryFrame);

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());

        static pclomp::NormalDistributionsTransform<PointType, PointType> ndt;
        pcl::PointCloud<PointType>::Ptr ndt_unused_result(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr ndt_closed_cloud(new pcl::PointCloud<PointType>());
        float transform_probability_ndt;

        // Chi tiet thuat toan lua chon
        if (loop_algoth == LoopType::ICP)
        {
            icp.setMaxCorrespondenceDistance(150); // use a value can cover 2*historyKeyframeSearchNum range in meter
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // ndt.setMaximumIterations(100);
            // ndt.setTransformationEpsilon(0.01);
            // ndt.setStepSize (0.1);
            // ndt.setResolution(1.0);
            // ndt.setRANSACIterations(0);

            // // Align clouds
            // ndt.setInputSource(cureKeyframeCloud);
            // ndt.setInputTarget(prevKeyframeCloud);
            // ndt.align(*ndt_unused_result);

            // Align clouds
            icp.setInputSource(cureKeyframeCloud);
            icp.setInputTarget(prevKeyframeCloud);
            // icp.align(*unused_result, ndt.getFinalTransformation());
            icp.align(*unused_result);
            // TODO icp align with initial

            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
                // publishCloud(&pubIcpKeyFrames, unused_result, timeLaserInfoStamp, odometryFrame);
            }
        }

        else if (loop_algoth == LoopType::NDT)
        {
            //Thong so cai dat
            ndt.setMaximumIterations(100);
            ndt.setTransformationEpsilon(1e-6);
            ndt.setResolution(2.0);
            ndt.setRANSACIterations(0);
            ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);

            // Align clouds
            ndt.setInputSource(cureKeyframeCloud);
            ndt.setInputTarget(prevKeyframeCloud);
            ndt.align(*ndt_unused_result);

            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::transformPointCloud(*cureKeyframeCloud, *ndt_closed_cloud, ndt.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, ndt_closed_cloud, timeLaserInfoStamp, odometryFrame);
            }

            transform_probability_ndt = ndt.getTransformationProbability();
            std::cout << "----------------------------------------------" << std::endl;
            std::cout <<  "Chi so matching map: " << transform_probability_ndt << std::endl;

            size_t kichthuoc_matching = ndt.getFinalNumIteration();
            std::cout <<  "Kich thuoc matching: " << kichthuoc_matching << std::endl;
        }

        else
        {
            ROS_ERROR_STREAM("Khong co loai thuat toan su dung: " << int(loop_algoth));
            ros::shutdown();
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        if (loop_algoth == LoopType::ICP)
        {
            correctionLidarFrame = icp.getFinalTransformation();
        }
        else if (loop_algoth == LoopType::NDT)
        {
            correctionLidarFrame = ndt.getFinalTransformation();
        }

        //Xac dinh gia tri noise cua matching Scancontext
        float noiseScore;
        if (loop_algoth == LoopType::ICP)
        {
            noiseScore = icp.getFitnessScore();

            if (icp.hasConverged() == false || noiseScore > historyKeyframeFitnessScore_SC)
            {
                std::cout << "ICP fitness test failed (" << noiseScore << " > " << historyKeyframeFitnessScore_SC << "). Reject this SC loop." << std::endl;
                return;
            }
            else
            {
                std::cout << "ICP fitness test passed (" << noiseScore << " < " << historyKeyframeFitnessScore_SC << "). Add this SC loop." << std::endl;
            }
        }

        else if (loop_algoth == LoopType::NDT)
        {
            noiseScore = ndt.getFitnessScore();

            if (ndt.hasConverged() == false || noiseScore > historyKeyframeFitnessScore_SC || transform_probability_ndt < 15.0)
            {
                std::cout << "NDT fitness test failed (" << noiseScore << " > " << historyKeyframeFitnessScore_SC << "). Reject this SC loop." << std::endl;
                return;
            }
            else
            {
                std::cout << "NDT fitness test passed (" << noiseScore << " < " << historyKeyframeFitnessScore_SC << "). Add this SC loop." << std::endl;
            }
        }

        // transform from world origin to wrong pose
        // Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        // Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);

        // pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
        // gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        // // gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        // gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        // gtsam::Vector robustNoiseVector6(6);
        // float robustNoiseScore = icp.getFitnessScore();
        // robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        // gtsam::noiseModel::Diagonal::shared_ptr robustConstraintNoise = gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6);

        pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        // robust kernel for a SC loop
        float robustNoiseScore = 0.5; // constant is ok...
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        gtsam::noiseModel::Base::shared_ptr robustConstraintNoise;
        robustConstraintNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::GemanMcClure::Create(1),            // optional: replacing Cauchy by DCS or GemanMcClure
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(robustConstraintNoise);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // multimap
    }       // performSCLoopClosure

    void performSCLoopClosure_intensity()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        int loopKeyCur, loopKeyPre;

        // find keys
        auto detectResult = scManager.detectLoopClosureID_intensity(); // first: nn index, second: yaw diff
        loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        loopKeyPre = detectResult;
        if (loopKeyPre == -1 /* No loop found */)
            return;

        std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, loopKeyPre); // giseop
        // loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

        // loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        // loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

        int base_key = 0;
        loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, base_key);
        loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key);

        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());

        static pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        pcl::PointCloud<PointType>::Ptr ndt_unused_result(new pcl::PointCloud<PointType>());
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius); // use a value can cover 2*historyKeyframeSearchNum range in meter
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        ndt.setMaximumIterations(100);
        ndt.setTransformationEpsilon(0.01);
        ndt.setStepSize (0.1);
        ndt.setResolution(1.0);
        ndt.setRANSACIterations(0);

        // Align clouds
        ndt.setInputSource(cureKeyframeCloud);
        ndt.setInputTarget(prevKeyframeCloud);
        ndt.align(*ndt_unused_result);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        icp.align(*unused_result, ndt.getFinalTransformation());
        // TODO icp align with initial

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore_SC)
        {
            std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore_SC << "). Reject this SC loop." << std::endl;
            return;
        }
        else
        {
            std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore_SC << "). Add this SC loop." << std::endl;
        }

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
            // publishCloud(&pubIcpKeyFrames, unused_result, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        // transform from world origin to wrong pose
        // Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        // Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);

        // pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
        // gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        // // gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        // gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        // gtsam::Vector robustNoiseVector6(6);
        // float robustNoiseScore = icp.getFitnessScore();
        // robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        // gtsam::noiseModel::Diagonal::shared_ptr robustConstraintNoise = gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6);

        pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        // robust kernel for a SC loop
        float robustNoiseScore = 0.5; // constant is ok...
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        gtsam::noiseModel::Base::shared_ptr robustConstraintNoise;
        robustConstraintNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::GemanMcClure::Create(1),            // optional: replacing Cauchy by DCS or GemanMcClure
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(robustConstraintNoise);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // multimap
    }       // performSCLoopClosure

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;                //Lấy phần tử ngay trước phần tử hiện tại đang kiểm tra Loop
        int loopKeyPre = -1;                                              //Key trống chưa tìm kiếm

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);                    //Kiểm tra trong Container đã có Key hiện tại
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop; // unused
        // for (int i = 0; i < (int)copy_cloudKeyPoses2D->size(); i++)
        //     copy_cloudKeyPoses2D->points[i].z = 1.1;              // __ 15/12/2021 Loại bỏ độ cao z

        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses2D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses2D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 20);

        //Check loop chỉ các Key thỏa mãn trong 1 khoảng thời gian so với điểm xét
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // std::cout << "i: " << i << ", " << "id: " << id << " " << "(squared distance: " << pointSearchSqDisLoop[i] << ")" << std::endl;
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;                                    //Key đã tìm thấy sự tương đồng để check loop
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())                          //Nếu chưa xuất hiện Loop nào thì bỏ qua bước check này
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];             //Lấy ra Marker của Loop đầu tiên
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();                         //Xóa phần tử đầu tiên trong tập Lôp

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)                                //Kích cỡ Point Cloud quá ít để check loop
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize)
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);

        //Octree filter
        // octreeDownsampling(nearKeyframes, cloud_temp, 0.2);

        *nearKeyframes = *cloud_temp;
    }

    //Phân ra kích thước submap vào vòng loop để test
    void loopFindNearKeyframesWithRespectTo(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum, const int _wrt_key)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize)    //Xét toàn bộ key từ điểm xét về trước và về sau
                continue;

            //nearKeyframes là toàn bộ pointcloud đã quét từ điểm xét về trước và từ điểm xét ra phía sau cần kiểm tra matching
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[_wrt_key]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[_wrt_key]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);

        //Octree filter
        // octreeDownsampling(nearKeyframes, cloud_temp, 0.2);

        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.scale.y = 0.1;
        markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization__ Khởi tạo góc của Key đầu tiên của map
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess //Dua theo IMU de tinh toan truoc goc quay tai vi tri dau tien
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)    //Kiem tra co Odom khong
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            }
            else
            {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation) --> Su dung IMU Incre de du doan pose ke tiep (chi voi goc quay)
        if (cloudInfo.imuAvailable == true)   //Kiem tra co IMU khong
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  //Ma trận quay
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;  //Tỉ lệ của dịch chuyển từ VT khởi tạo -> hiện tại

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;        //Ma trận hiện tại * Ma trận tỉ lệ dich chuyển
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();  //Kích thước các key hiện tại trong map
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]); //Đẩy các point thỏa mãn kích thước Key Frame
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        //Them ngay 23/02/2021
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)  //Duyệt qua tất cả các điểm thỏa mãn giới hạn Key Frame
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)   //Kiểm tra khoảng cách của điểm đã extract và cloudKey trong bán kính cho phép thì thêm vào map
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;

            //Kiểm tra intensity có trùng vùng không
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())   //hàm find trong std::map để tìm 1 phần tử trong map tại vị trí thisKeyInd
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;    //Giá trị intensity đầu tiên là corner
                *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;     //Giá trị intensity thứ hai là surf
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);   //Gasn 2 giá trị này vào Container
            }
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 10000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        //Khi biến xác định vòng lặp == true, khung vòng lặp sẽ được tách ra, hoặc point lân cận được tách ra
        if (loopClosureEnableFlag == true)
        {
            extractForLoopClosure();
        } else {
            extractNearby();
        }

        // extractNearby();
    }

    void downsampleCurrentScan()
    {
        //---------------------------------------------------------------
        laserCloudRawDS->clear();
        downSizeFilterSC.setInputCloud(laserCloudRaw);
        downSizeFilterSC.filter(*laserCloudRawDS);
        // pcl::copyPointCloud(*laserCloudRaw, *laserCloudRawDS);
        //--------------------------------------------------------------
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();        //Số lượng điểm, Corner

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();            //Số lượng điểm Surf
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        //Duyệt các điểm trong Corner và Surf, xây dựng ràng buộc
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            //test

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            if (pointSearchSqDis[4] < 1.0)
            {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1)
                    {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0)
            {
                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1)
                    {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            if (laserCloudOriCornerFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            if (laserCloudOriSurfFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50)
        {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++)
        {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0)
        {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }
                else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05)
        {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // Kiểm tra số lượng điểm góc và mặt phẳng
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;
            }

            transformUpdate();
        }
        else
        {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    //Bien giup xac dinh co them key moi vao map tao khong
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());  //Lấy ma trận của key cuối cùng đang xét
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;       //Tính ra ma trận chuyển giữa key cuối cùng và key sắp tạo
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);     //Lấy rã, y, z của ma trận chuyển để lấy khoảng cách

        //Dieu kien kiem tra neu goc roll, pitch, yaw va khoang cach tu diem key moi den key cu khong toi thieu bang Thresh hold thi loai bo)
        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

//Them odom cho IMU, lidar
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // Toi uu hoa diem initial du doan
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));    //Thêm mối liên hệ giữ VT khởi đàu (0, 0, 0) -> X1
            writeVertex(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
            gtsam::Pose3 relPose = poseFrom.between(poseTo);
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), relPose, odometryNoise));   //Thêm mối liên hệ giữa pose X1 -> X2
            // Toi uu hoa diem initial du doan
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            writeVertex(cloudKeyPoses3D->size(), poseTo);
            writeEdge({cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size()}, relPose); // giseop
        }
    }

//Tìm tọa độ gốc 0 của map
    void zeroUTM()
    {
        if (gpsTopicQueue.empty())
            return;

        while (!gpsTopicQueue.empty())
        {
            if (gpsTopicQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsTopicQueue.pop_front();
            }
            else if (gpsTopicQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                //--------------Test 1/6/2021 cho toa do dia cau
                geographic_msgs::GeoPointStamped thisTopicGPS = gpsTopicQueue.front();

                if(map_project == projectmapType::UTM)
                {
                    geodesy::UTMPoint gps_utm;
                    geodesy::fromMsg(thisTopicGPS.position, gps_utm);

                    Eigen::Vector3d xyz_utm(gps_utm.easting, gps_utm.northing, 0.0);

                    if(!zero_projector)
                    {
                        zero_projector = xyz_utm;
                    }
                    break;
                }

                else if(map_project == projectmapType::MGRS)
                {
                    lanelet::GPSPoint gps_point;
                    lanelet::projection::MGRSProjector projector;

                    gps_point.lat = thisTopicGPS.position.latitude;
                    gps_point.lon = thisTopicGPS.position.longitude;
                    gps_point.ele = 0.0;

                    // if (!useGpsElevation)      //Nếu độ cao GPS kém
                    // {
                    //     gps_point.ele = 0.0;
                    // }

                    lanelet::BasicPoint3d gps_mgrs = projector.forward(gps_point);
                    Eigen::Vector3d xyz_mgrs(gps_mgrs.x(), gps_mgrs.y(), gps_mgrs.z());

                    if(!zero_projector)
                    {
                        zero_projector = xyz_mgrs;
                    }
                    break;
                }else
                {
                    ROS_ERROR_STREAM("Khong chon phep chieu phu hop: " << int(map_project));
                    ros::shutdown();
                }
            }
        }

        // cout << "GPS_X:" << zero_projector->x() << endl;
        // cout << "GPS_Y:" << zero_projector->y() << endl;
        // cout << "GPS_Z:" << zero_projector->z() << endl;
    }

// Them cac odom gps vao map
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 2.0)    //So sanh nếu số lượng Key của bản đồ tạo quá it và khoảng cách tối thiểu là  m thì mới xử lý Key GPS
                return;
        }

        // Sai lech odom GPS theo trục X và Y nhỏ -> sai lệch X < delta và sai lệch Y < delta
        if ((poseCovariance(3,3) < poseCovThreshold) && (poseCovariance(4,4) < poseCovThreshold))
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            //Kiem tra dong bo thoi gian GPS va lidar quet bang nhau khong
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else if (aLoopIsClosed == true)             //Nếu trong trường hợp mà xuất hiện Loop thì ưu tiên Loop, bỏ GPS
            {
                break;
                std::cout << "Ưu tiên check loop" << std::endl;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];

                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)    //Nhiễu GPS quá thì bỏ qua ko check
                     continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z;

                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }
                else
                {
                    if (noise_z > gpsCovThreshold_z)
                    {
                        gps_z = transformTobeMapped[5];
                        noise_z = 0.01;
                    }
                    else if (noise_z < gpsCovThreshold_z)
                    {
                        gps_z = thisGPS.pose.pose.position.z;
                        std::cout << "fix_zGPS" << std::endl;
                    }
                }

                // TEST
                // GPS not properly initialized (0,0,0)
                if ((abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6) || fabs(gps_x) > 100000.0 || fabs(gps_y) > 100000.0)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;

                if (pointDistance(curGPSPoint, lastGPSPoint) < 2.0)    //Tối thiểu 3 m thì mới thêm Key GPS để tối ưu Graph
                    continue;
                else
                    lastGPSPoint = curGPSPoint;
                    std::cout << "fix_xyGPS" << std::endl;

                // (GPS factor cua GTSAM)
                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aGPSIsClosed = true;     //Bien cho phep them GPS factor
                break;
            }
        }
    }

//Kiem tra va bat vong lap point
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            auto noiseBetween = loopNoiseQueue[i]; // giseop for polymorhpism // shared_ptr<gtsam::noiseModel::Base>, typedef noiseModel::Base::shared_ptr gtsam::SharedNoiseModel
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // Zero UTM
        zeroUTM();

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor(); // radius search loop factor (I changed the orignal func name addLoopFactor to addLoopFactor)

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);     //Khởi tạo điểm đầu
        isam->update();        //Odom first factor

        if (aGPSIsClosed == true)
        {
            isam->update();    //GPS factor
            isam->update();
        }
        if (aLoopIsClosed == true)
        {
            isam->update();    //Loop factor
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        gtsam::Pose3 latestEstimate;     //Tính toán ước tính

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time  = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // cout << "****************************************************" << endl;
        // cout << "Sai lệch vị trí sau tối ưu: " << endl;
        // cout << "X: " << poseCovariance(3,3) << endl;
        // cout << "Y: " << poseCovariance(4,4) << endl;
        // cout << "Z: " << poseCovariance(5,5) << endl;

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame_raw(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame_raw(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame_raw);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame_raw);

        octreeDownsampling(thisCornerKeyFrame_raw, thisCornerKeyFrame, kDownsampleVoxelSize);
        octreeDownsampling(thisSurfKeyFrame_raw, thisSurfKeyFrame, kDownsampleVoxelSize);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        //-----------------------------------------------------------------------------------------------------------------------------
        // Scan Context loop detector - giseop
        // - SINGLE_SCAN_FULL: using downsampled original point cloud (/full_cloud_projected + downsampling)
        // - SINGLE_SCAN_FEAT: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
        // - MULTI_SCAN_FEAT: using NearKeyframes (because a MulRan scan does not have beyond region, so to solve this issue ... )
        const SCInputType sc_input_type = SCInputType::SINGLE_SCAN_FULL; // change this
        scManager.init_color();   //Thêm ngày 11/10/2021

        if (sc_input_type == SCInputType::SINGLE_SCAN_FULL)
        {
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudRawDS, *thisRawCloudKeyFrame);
            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
        }
        else if (sc_input_type == SCInputType::SINGLE_SCAN_FEAT)
        {
            scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame);
        }
        else if (sc_input_type == SCInputType::MULTI_SCAN_FEAT)
        {
            pcl::PointCloud<PointType>::Ptr multiKeyFrameFeatureCloud(new pcl::PointCloud<PointType>());
            loopFindNearKeyframes(multiKeyFrameFeatureCloud, cloudKeyPoses6D->size() - 1, historyKeyframeSearchNum);
            scManager.makeAndSaveScancontextAndKeys(*multiKeyFrameFeatureCloud);
        }

        // save sc data
        const auto &curr_scd = scManager.getConstRefRecentSCD();
        std::string curr_scd_node_idx = padZeros(scManager.polarcontexts_.size() - 1);

        saveSCD(saveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);

        //Xuất Scancontext ra file ảnh
        cv_bridge::CvImage out_msg;
        out_msg.header.frame_id  = odometryFrame;
        out_msg.header.stamp  = timeLaserInfoStamp;
        out_msg.encoding = sensor_msgs::image_encodings::RGB8;
        out_msg.image    = scManager.getLastISCRGB_PC();
        image_sc_pub.publish(out_msg.toImageMsg());
        //----------------------------------------------------------

        // save keyframe cloud as file giseop
        bool saveRawCloud{true};
        pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());

        if (saveRawCloud)
        {
            *thisKeyFrameCloud += *laserCloudRaw;
        }
        else
        {
            *thisKeyFrameCloud += *thisCornerKeyFrame;
            *thisKeyFrameCloud += *thisSurfKeyFrame;
        }

        pcl::io::savePCDFileBinary(saveNodePCDDirectory + curr_scd_node_idx + ".pcd", *thisKeyFrameCloud);
        saveBINfile(saveNodeBINDirectory + curr_scd_node_idx + ".bin", thisKeyFrameCloud);

        pgTimeSaveStream << laserCloudRawTime << std::endl;
        //----------------------------------------------------------------------------------------------------------

        // save path for visualization
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if ((aLoopIsClosed == true) || (aGPSIsClosed == true))
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // Cap nhat key pose moi
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
            aGPSIsClosed  = false;
        }
    }

    void updatePath(const PointTypePose &pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

//Tao odometry de theo doi path cua vat the di chuyen
    void publishOdometry()
    {
        // Publish odometry for ROS (global) __ Odom Lidar
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);

        // std::cout << "----------------------------------------------------------------------" << std::endl;
        // std::cout << "transformtobemap: " << transformTobeMapped[0] << ", " << transformTobeMapped[1] << ", " << transformTobeMapped[2] << ", " << transformTobeMapped[3] << ", " << transformTobeMapped[4] << ", " << transformTobeMapped[5] << std::endl;

        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");      //frame_id: Odom, child: lidar_link
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental) __ Odom lidar increment
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        }
        else
        {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);

            // std::cout << "----------------------------------------------------------------------" << std::endl;
            // std::cout << "increOdomAffine: " << x << ", " << y << ", " << z << ", " << roll << ", " << pitch << ", " << yaw << std::endl;

            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapping_pcd");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Buoc 4.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();   //Vong lap chuong trinh tu diem nay

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}