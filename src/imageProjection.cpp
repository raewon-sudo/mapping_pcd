#include "utility.h"
#include "mapping_pcd/cloud_info.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

struct PandarPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PandarPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (double, timestamp, timestamp)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

//--------------------------------- Image point ---------------------------------------------
sensor_msgs::ImagePtr cvmat2msg(const cv::Mat &_img)
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _img).toImageMsg();
  return msg;
}

std::pair<int, int> resetRimgSize(const std::pair<float, float> _fov, const float _resize_ratio)
{
    // default is 1 deg x 1 deg
    float alpha_vfov = _resize_ratio;
    float alpha_hfov = _resize_ratio;

    float V_FOV = _fov.first;
    float H_FOV = _fov.second;

    int NUM_RANGE_IMG_ROW = std::round(V_FOV*alpha_vfov);
    int NUM_RANGE_IMG_COL = std::round(H_FOV*alpha_hfov);

    std::pair<int, int> rimg {NUM_RANGE_IMG_ROW, NUM_RANGE_IMG_COL};
    return rimg;
}

cv::Mat convertColorMappedImg (const cv::Mat &_src, std::pair<float, float> _caxis)
{
  float min_color_val = _caxis.first;
  float max_color_val = _caxis.second;

  cv::Mat image_dst;
  image_dst = 255 * (_src - min_color_val) / (max_color_val - min_color_val);
  image_dst.convertTo(image_dst, CV_8UC1);                                            //Chuyển ma trận thành dạng 8 bit màu có 1 kênh từ 0 -> 255

  cv::applyColorMap(image_dst, image_dst, cv::COLORMAP_JET);                          // Gán trạng thái màu full dải lên ma trận ảnh chưa có màu

  return image_dst;
}

void pubRangeImg(cv::Mat& _rimg,
                sensor_msgs::ImagePtr& _msg,
                ros::Publisher& _publiser,
                std::pair<float, float> _caxis)
{
    cv::Mat scan_rimg_viz = convertColorMappedImg(_rimg, _caxis);
    _msg = cvmat2msg(scan_rimg_viz);
    _publiser.publish(_msg);
} // pubRangeImg
//-------------------------------------------------------------------------------------------------------

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;

    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;
    ros::Publisher image_point_pub;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointType>::Ptr laserCloudIn_noRing;
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PandarPointXYZIRT>::Ptr tmpPandarCloudIn;

    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    mapping_pcd::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanNext;
    std_msgs::Header cloudHeader;
    int cloudSize;
    int count;

public:
    ImageProjection():
    deskewFlag(0)
    {
        //Odom được lấy bằng cách kết hợp GPS + IMU thông qua robot_localization
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("mapping_pcd/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<mapping_pcd::cloud_info> ("mapping_pcd/deskew/cloud_info", 1);
        image_point_pub   = nh.advertise<sensor_msgs::Image>("/image_point", 10);

        //Khởi tạo
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn_noRing.reset(new pcl::PointCloud<PointType>());
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        tmpPandarCloudIn.reset(new pcl::PointCloud<PandarPointXYZIRT>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn_noRing->clear();
        laserCloudIn->clear();
        extractedCloud->clear();

        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));     //Set tat ca pixel tren ma tran anh tahnh FLT_MAX

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
    {
        //Chuyển dữ liệu IMU gốc sang hệ trục tọa độ Lidar __ IMU frame --> Lidar frame
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x <<
        //       ", y: " << thisImu.linear_acceleration.y <<
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x <<
        //       ", y: " << thisImu.angular_velocity.y <<
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg)
    {
        // Đẩy dữ liệu Odom vào hàng đợi Odom
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        //Xóa các điểm Nan point và kiểm tra time và ring
        if (!cachePointCloud(laserCloudMsg))
            return;

        //Loại bỏ méo của IMU và Lidar.
        //Tìm dữ liệu Lidar, Odom, IMU+GPS trong mỗi khoảng thời gian quét
        //Tính toán acc linear và acc rotate
        if (!deskewInfo())
            return;

        //Điểm đã được loại bỏ các thành phần lỗi
        //Chiếu lại đám mây điểm về dạng 1800x16 -> Khử biến dạng
        projectPointCloud();

        pub_image_point();

        //Lấy ra đám mây điểm từ phép chiếu để đo độ sâu các điểm ở featureExtraction
        cloudExtraction();

        // detection_curb();

        //Xuất bản ra bản tin Point "cloud info"
        publishClouds();

        //Đặt lại các thông số
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // lấy dữ liệu từ Topic point đẩy sang hàng đợi Lidar
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        //Chuyển hàng đợi point mới nhất của vòng quét vào currentCloudMsg
        currentCloudMsg = std::move(cloudQueue.front());

        // Lấy dữ liệu thời gian mới nhất
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();

        cloudQueue.pop_front();  //Xóa dữ liệu hàng đợi mới nhất vừa lấy ra

        //Loại bỏ các Nan point
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn_noRing);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn_noRing, *laserCloudIn_noRing, indices);

        if (useCloudRing == true)
        {
            // Tùy loại Lidar để quy đổi dữ liệu đồng bộ thời gian về giống Velodyne Lidar
            if (sensor == SensorType::VELODYNE)
            {
                pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
            }
            else if (sensor == SensorType::OUSTER)
            {
                // Convert to Velodyne format
                pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
                laserCloudIn->points.resize(tmpOusterCloudIn->size());
                laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
                for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
                {
                    auto &src = tmpOusterCloudIn->points[i];
                    auto &dst = laserCloudIn->points[i];
                    dst.x = src.x;
                    dst.y = src.y;
                    dst.z = src.z;
                    dst.intensity = src.intensity;
                    dst.ring = src.ring;
                    dst.time = src.t * 1e-9f;
                }
            }
            else if (sensor == SensorType::PANDAR)
            {
                // Convert to Velodyne format
                pcl::moveFromROSMsg(currentCloudMsg, *tmpPandarCloudIn);
                laserCloudIn->points.resize(tmpPandarCloudIn->size());
                laserCloudIn->is_dense = tmpPandarCloudIn->is_dense;
                for (size_t i = 0; i < tmpPandarCloudIn->size(); i++)
                {
                    auto &src = tmpPandarCloudIn->points[i];
                    auto &dst = laserCloudIn->points[i];
                    dst.x = src.x;
                    dst.y = src.y;
                    dst.z = src.z;
                    dst.intensity = src.intensity;
                    dst.ring = src.ring;
                    dst.time = float(src.timestamp);
                }
            }
        }
        else
        {
            // convert cloud
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }

        timeScanNext = timeScanCur + laserCloudIn->points.back().time;
        // timeScanNext = cloudQueue.front().header.stamp.toSec();  //Lấy thời gian quét cuối là thời gian bắt đầu hàng đợi mới nhất kế tiếp (Đã loại bỏ hàng đợi đầu đã lấy ra)

        //Kiểm tra mật độ điểm với Lidar có Ring
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

            // check ring channel __ TEST_06_07_2021
            // static int ringFlag = 0;
            // if (ringFlag == 0)
            // {
            //     ringFlag = -1;
            //     for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            //     {
            //         if (currentCloudMsg.fields[i].name == "ring")
            //         {
            //             ringFlag = 1;
            //             break;
            //         }
            //     }
            //     if (ringFlag == -1)
            //     {
            //         ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
            //         ros::shutdown();
            //     }
            // }

            // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t" || field.name == "timestamp")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                    ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }
        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // Đảm bảo có dữ liệu IMU được thu nhận và thời gian quét nằm trong khoảng thời gian hoạt động của Lidar (giữa 2 hàng đợi liên tiếp của Lidar)
        // if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext - laserCloudIn->points.back().time)
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");

            return false;
        }

        //Kiểm tra dữ liệu IMU, tính toán dữ liệu IMU tương ứng với Lidar Frame bao gồm tịnh tiến và góc quay
        //Cấu hình thông số IMU
        imuDeskewInfo();

        //Cấu hình thông số Odom
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;      //Dùng trong giai đoạn tối ưu hóa bản đồ, nên ban đầu hãy đặt = false

        //Kiểm tra loại bỏ sai lệch của IMU
        while (!imuQueue.empty())    //Check thời gian quét imu
        {
            //Kiểm tra thời gian của hàng đợi đầu tiên của IMU, lấy ngưỡng 0.01 để bỏ dữ liệu IMU cũ hơn
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();   //Xóa bỏ hàng đợi này của IMU
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)  //Kiểm tra toàn bộ hàng đợi IMU
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();  //Thơi gian của hàng đợi IMU hiện tại

            // get roll, pitch, and yaw estimation for this scan   //Lấy R, P, Y cho lần quét đầu tiên (VT IMU theo góc hiện tại)
            if (currentImuTime <= timeScanCur)
                //Sử dụng tính toán góc Euler để ước tính khởi tạo RPY quét và cung cấp gia trị này cho CloudInfo để dùng trong phần tối ưu hóa bản đồ
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);


            if (currentImuTime > timeScanNext + 0.01)  //Nếu thời gian của IMU hiện tại > Thời gian cuôi 1 lần quét của Lidar + 0.01 thì thoat ra, để đảm bảo không xét hàng đợi IMU mà ko đồng bộ thời gian với Lidar
                break;

            //Các giá trị khai báo sau đều = 0 tại lần khởi tạo đầu tiên
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;  //Đẩy giá trị xét lên 1
                continue;
            }  //Bỏ gia trị IMU đầu tiên -> con trỏ 0

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);   //Lấy giá trị IMU để tách ra các angular velocity theo X, Y, Z

            //integrate rotation -> Góc tại điểm imu tiêp theo
            //Tính toán angular velocity với thời gian để khử biến dạng cho IMU kế tiếp theo vận tốc góc
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];  //Khoảng thời gian giữa 2 lần quét IMU liên tiếp
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;

        }     //Góc hiện tại của IMU = Góc trước + t * Vận tốc góc

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        //IMU này đã được tiền xử lý trước
        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        //Đánh dấu dữ liệu thông tin Odom được xử lý là hợp lệ
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan -- Lấy điểm bắt đầu quét cho bản đồ
        nav_msgs::Odometry startOdomMsg;

        //Duyệt hàng đợi Odom và dùng Odom cho thông tin tư thế đám mây điểm
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        //Lấy R, P, Y của Odom
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);  //Đẩy giá trị Odom ra orientation
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);  //Lấy roll, pitch, yaw từ giá trị orien: x, y, z, w

        // Initial guess used in mapOptimization -> Khởi tạo dự đoán Gauss sẽ dùng trong mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;

        //Odom ở cuối quá trình quét
        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }

        //Ma trận phương sai covariance, nếu covariance ko nhất quán ở Odom đầu và cuối thì thoát ra
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        //Ma trận vị trí Odom ban đầu
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x,
                                                            startOdomMsg.pose.pose.position.y,
                                                            startOdomMsg.pose.pose.position.z,
                                                            roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        //Lấy ma trận vị trí của Odom cuối
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x,
                                                          endOdomMsg.pose.pose.position.y,
                                                          endOdomMsg.pose.pose.position.z,
                                                          roll, pitch, yaw);

        //Ma trận chuyển đổi giữa TG bắt đầu và TG kết thúc
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;   //->TranEnd/TranBegin

        float rollIncre, pitchIncre, yawIncre;
        //Tính độ dịch chuyển và tăng góc quay theo các trục
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        // Point time hiện tại nằm trước IMU time, loại bỏ
        // Đảm bảo TG point nằm giữa 2 khung dữ liệu IMU
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        //point time nằm ở điểm bắt đầu của hàng đợi IMU  và được yêu cầu ở imuDeskewInfo
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            //Theo thông tin TG point,
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        //Giá trị gia tăng của tọa độ nếu đi chuyển tốc độ cao mới đáng kể
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.
        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        //Kiểm tra nếu có biến time thì điều chỉnh mỗi điểm méo
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        //Thời gian thực của điểm trong frame hiện tại
        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        //Chiếu đám mây điểm theo đường xuyên tâm
        float range, verticalAngle, horizonAngle;
        int rowIdn, columnIdn, index;

        cloudSize = laserCloudIn_noRing->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;

            if (sensor == SensorType::VELODYNE)
            {
                thisPoint.x = laserCloudIn_noRing->points[i].x;
                thisPoint.y = laserCloudIn_noRing->points[i].y;
                thisPoint.z = laserCloudIn_noRing->points[i].z;
                thisPoint.intensity = laserCloudIn_noRing->points[i].intensity;
            }
            else if (sensor == SensorType::PANDAR)
            {
                //point raw nguoc chieu so voi khai bao
                thisPoint.x = - laserCloudIn_noRing->points[i].x;
                thisPoint.y = - laserCloudIn_noRing->points[i].y;
                thisPoint.z = laserCloudIn_noRing->points[i].z;
                thisPoint.intensity = laserCloudIn_noRing->points[i].intensity;
            }

            range = pointDistance(thisPoint);   //Tra ve gia tri su dung cho Ma tran rangeMat cua phep chieu Ma tran anh CV cho pointcloud chua sap xep
            //range = căn bậc 2((thisPoint.x)^2+(thisPoint.y)^2+(thisPoint.z)^2)

            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            // Xác định row và column của ma trận chiếu các điểm
            if (useCloudRing == true){
                rowIdn = laserCloudIn->points[i].ring;
            }
            else{
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;      //atan2 = arctan cua gia tri chuyen thanh góc của tất cả các góc phần tư
            // horizonAngle: Goc theo do (Tam nhin FOV ngang)

            //Độ phân giải góc
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // thisPoint = deskewPoint(&thisPoint, currentCloudMsg.header.stamp.toSec());

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            index = columnIdn + rowIdn * Horizon_SCAN;           //Cac dinh thuoc vong quet lidar
            fullCloud->points[index] = thisPoint;
        }
    }

    void pub_image_point()
    {
        sensor_msgs::ImagePtr matrix_point_msg;
        float rimg_color_min = 0.0;
        float rimg_color_max = 10.0;
        std::pair<float, float> kRangeColorAxis = std::pair<float, float> {rimg_color_min, rimg_color_max}; // meter
        float kVFOV = 40.0;
        float kHFOV = 360.0;
        std::pair<float, float> kFOV = std::pair<float, float>(kVFOV, kHFOV);
        float _res_alpha = 1 / ang_res_y;

        std::pair<int, int> rimg_shape = resetRimgSize(kFOV, _res_alpha);

        float deg_per_pixel = 1.0 / _res_alpha;
        // ROS_INFO_STREAM("\033[1;32m Start image starts with resolution: x" << _res_alpha << " (" << deg_per_pixel << " deg/pixel)\033[0m");
        // ROS_INFO_STREAM("\033[1;32m -- The range image size is: [" << rimg_shape.first << ", " << rimg_shape.second << "].\033[0m");
        // ROS_INFO_STREAM("\033[1;32m -- The number of map points: " << laserCloudIn_noRing->points.size() << "\033[0m");

        cv::Mat matrix_point = cv::Mat(rimg_shape.first, rimg_shape.second, CV_32FC1, cv::Scalar::all(10000.0));

        cloudSize = laserCloudIn_noRing->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;

            if (sensor == SensorType::VELODYNE)
            {
                thisPoint.x = laserCloudIn_noRing->points[i].x;
                thisPoint.y = laserCloudIn_noRing->points[i].y;
                thisPoint.z = laserCloudIn_noRing->points[i].z;
                thisPoint.intensity = laserCloudIn_noRing->points[i].intensity;
            }
            else if (sensor == SensorType::PANDAR)
            {
                //point raw nguoc chieu so voi khai bao
                thisPoint.x = - laserCloudIn_noRing->points[i].x;
                thisPoint.y = - laserCloudIn_noRing->points[i].y;
                thisPoint.z = laserCloudIn_noRing->points[i].z;
                thisPoint.intensity = laserCloudIn_noRing->points[i].intensity;
            }

            float range = pointDistance(thisPoint);
            float pitch = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            float yaw   = atan2(thisPoint.y, thisPoint.x) * 180 / M_PI;

            int lower_bound_row_idx {0};
            int lower_bound_col_idx {0};
            int upper_bound_row_idx {rimg_shape.first - 1};
            int upper_bound_col_idx {rimg_shape.second - 1};

            int u = int(std::min(std::max(float(std::round(rimg_shape.first * (1 - (pitch + ang_bottom)/(ang_bottom + 15.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
            int v = int(std::min(std::max(float(std::round(rimg_shape.second * (0.5 * ((yaw / 180)+1)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

            if (range < matrix_point.at<float>(u, v)) {
                matrix_point.at<float>(u, v) = range;
            }
        }

        pubRangeImg(matrix_point, matrix_point_msg, image_point_pub, kRangeColorAxis);
    }

    void cloudExtraction()
    {
        //Trich các điểm
        count = 0;
        // extract segmented cloud for lidar odometry__Trích xuất các thành phần của Lidar
        for (int i = 0; i < N_SCAN; ++i)      //N_SCAN: So kenh Lidar (Velodyne: 16, 32, 64 || Livox: 6)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;
            //Kiêm tra độ cong của mỗi point xác định chỉ được tính trong cùng 1 lần quét
            // Nên luôn phải loại bỏ 5 điểm trước và sau mỗi lần quét

            for (int j = 0; j < Horizon_SCAN; ++j)     //Horizon_SCAN: So cung nam tren 1 vong quet (Velodyne: 1800 || Livox: 4000)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    //Đánh dấu vị trí cột của điểm để check ở featureExtraction
                    cloudInfo.pointColInd[count] = j;   //j = 1 -> 1800 với mỗi Line quét
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);  //Khoảng cách/Độ sâu truwofng điểm tới vị trí gốc O của Lidar
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }

    //18/10/2021
    // void detection_curb()
    // {
        // //Khai báo biến
        // pcl::KdTreeFLANN<PointType>::Ptr point_check;
        // pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
        // PointType thisPoint;
        // std::vector<int> pointIdxRadiusSearch;
        // std::vector<float> pointRadiusSquaredDistance;
        // int cloudSize_curb;
        // float radius = 0.25;
        // int k =10;
        // std::vector<float> avg, fix, weight;

        // //Xử lý
        // cloudSize_curb = laserCloudIn_noRing->points.size();
        // // range image projection
        // for (int i = 0; i < cloudSize_curb; ++i)
        // {
        //     thisPoint.x = laserCloudIn_noRing->points[i].x;
        //     thisPoint.y = laserCloudIn_noRing->points[i].y;
        //     thisPoint.z = laserCloudIn_noRing->points[i].z;
        //     thisPoint.intensity = laserCloudIn_noRing->points[i].intensity;

        //     point_check->radiusSearch(thisPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        //     for int j = 0;j < pointIdxRadiusSearch.size();j++)      //k gia tri: pointIdxRadiusSearch.size() -1
        //     {

        //        (*cloud)avg = (*cloud)avg + (*cloud)[pointIdxRadiusSearch[j]].x;
        //        (*cloud)avg = (*cloud)avg + (*cloud)[pointIdxRadiusSearch[j]].y;
        //        (*cloud)avg = (*cloud)avg + (*cloud)[pointIdxRadiusSearch[j]].z;

        //     }

        //     avg = avg / (pointIdxRadiusSearch.size() -1);

        //     for int k = 0;k < pointIdxRadiusSearch.size();k++)      //k gia tri: pointIdxRadiusSearch.size() -1
        //     {

        //        fix = fix + fabs(avg - (*cloud)[pointIdxRadiusSearch[k]].x);
        //     }

        //     fix = fix / (pointIdxRadiusSearch.size() -1);                //Giá trị muy = (1/k)*(Tổng xích ma theo (i=1->k) của |avg_p - pi|)

        //     weight = std::exp(-((thisPoint - )*(thisPoint - ))/(fix*fix));
        // }

    //     pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;
    //     normalEstimation.setInputCloud (*laserCloudIn_noRing);
    //     pcl::search::KdTree<PointType>::Ptr point_check (new pcl::search::KdTree<PointType>);
    //     normalEstimation.setSearchMethod (point_check);

    //     // Output datasets
    //     pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    //     // Use all neighbors in a sphere of radius 3cm
    //     normalEstimation.setRadiusSearch (0.25);

    //     // Compute the features
    //     normalEstimation.compute (*cloud_normals);

    //     // cloud_normals->points.size () should have the same size as the input cloud->points.size ()
    //     std::cout << "cloud_normals->points.size (): " << cloud_normals->points.size () << std::endl;

    // }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapping_pcd");

    ImageProjection IP;

    ROS_INFO("\033[1;32m----> Buoc 1.\033[0m");
    ros::MultiThreadedSpinner spinner(3);   //Xử lý đa luồng
    spinner.spin();

    return 0;
}
