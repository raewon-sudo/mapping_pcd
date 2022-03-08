#include "utility.h"
#include "mapping_pcd/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    mapping_pcd::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<mapping_pcd::cloud_info>("mapping_pcd/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<mapping_pcd::cloud_info> ("mapping_pcd/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("mapping_pcd/feature/cloud_surface", 1);
        
        initializationValue();
    }

    void initializationValue()
    {
        if (sensor != SensorType::LIVOX_HORIZON)
        {
            cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        }else if (sensor == SensorType::LIVOX_HORIZON)
        {
            cloudSmoothness.resize(400000);
        }

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const mapping_pcd::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

// Tinh toán độ cong của tập hợp điểm point
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        //Tính toán độ cong: Sử dụng 5 điểm trước và sau mỗi 
        // điểm để tính độ cong của point 
        // vì thế 5 điểm đầu tiên và 5 điểm cuối cùng được bỏ qua
        for (int i = 5; i < cloudSize - 5; i++) 
        {
            //pointRange là 1 ma trận của [x, y, z] thể hiện vị trí của Point trong không gian với mốc 0 Lidar; 
            //diffRange là ma trận dọc của [diffX, diffY, diffZ]
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - 10 * cloudInfo.pointRange[i]
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            
            
            //Theo Lý thuyết LOAM phép toán thực tế với hệ tọa độ Point(x, y, z) khai triển từ phép toán trên là:
            // float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
            //             + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
            //             + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
            //             + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
            //             + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
            //             + laserCloud->points[i + 5].x;
            // float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
            //             + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
            //             + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
            //             + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
            //             + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
            //             + laserCloud->points[i + 5].y;
            // float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
            //             + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
            //             + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
            //             + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
            //             + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
            //             + laserCloud->points[i + 5].z;

            //Tính toán độ cong
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            
            //Khai báo ban đầu, các điểm chưa được chọn
            cloudNeighborPicked[i] = 0;
            //Khởi tạo điểm Others (lý thuyết LOAM -> Các điểm ít phẳng hơn)
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];  //Độ cong được lưu vào biến CloudSmoothess
            cloudSmoothness[i].ind = i;     //Lưu lại chỉ số của điểm cong đang xét là i
        }
    }

//Đánh dấu các điểm xấu để loại bỏ
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        //Các điểm kiểm tra có thể bị chặn vởi độ nghiêng của điểm thuộc plane //với chùm tia
        // Hoặc điểm ngoại lệ/bị che khuất. Sự khác biệt giữ điểm i và điểm tiếp theo i+1 
        // do vậy cần - 6 điểm cần xét tính từ cuối lần quét
        for (int i = 5; i < cloudSize - 6; ++i)  
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10){
                // 10 pixel diff in range image
                //Khoảng cách tối thiểu cần check giữa 2 điểm là 10 pixel/1800 pixel
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1; //Điểm đã được lọai ra
                }else if (depth2 - depth1 > 0.3){     //Yêu cầu tối thiếu độ sâu trường giữa 2 điểm liên tiếp cần > 0.3 (> 0.1 theo lý thuyết LOAM)
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1; //Điểm đã được lọai ra
                }
            }
            // parallel beam //Loai bo cac diem thuoc chum tia song song(Tuc la loai bo cac diem thuoc mat phang // chum laze lidar)
            //Theo lý thuyết LOAM công thức thực tế là:
            // float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
            // float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
            // float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
            // float columnDiff = diffX * diffX + diffY * diffY + diffZ * diffZ; -> DO chenh tro 1 line
            
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            //Các điểm này được coi là điểm ngoại lai, bao gồm các điểm trên sườn dốc, các điểm lồi lõm mạnh và các điểm nhất định trong khu vực mở. 
            // Chúng được tính là đã lọc và bị loại bỏ
            //Theo lý thuyết LOAM có thể tới 0.002 với diff1 là: khcach(i-1, i); diff2 là: khcach(i, i+1)
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;  //Điểm đã được loại ra
        }
    }

// Trich xuất các dạng đối tượng surf hoặc corner trong Point
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        
        //Chia các điểm trên mỗi line thành 2 loại: Edge và Plane
        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)  //Kiem tra tren toan bo Line quet cua Lidar (VD Lidar Velodyne 16 -> N_SCAN = 16)
        {
            surfaceCloudScan->clear();

            //Chia đường cong trên môi line quét thành kNumRegion =6 phân bằng nhau để đảm bảo các điểm xung quanh được chọn là đặc trưng
            for (int j = 0; j < 6; j++)
            {
                //Phần bắt đầu xét: sp = scanStartInd + (scanEndInd - scanStartInd) * j / 6
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                //Phần cuối xét: ep = (scanStartInd - 1) + (scanEndInd - scanStartInd) * (j+1) / 6
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)   //Nếu vùng xét bắt đầu > Vùng xét kết thúc thì bỏ qua
                    continue;

                //Hàm sắp xếp từ nhỏ -> lớn cho độ cong
                //Sắp xếp độ cong từ nhỏ -> lớn. Theo lý thuyết LOAM: for(k=sp+1; k<=ep; k++)
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());
                //Cach khac
                // for (int k = sp + 1; k <= ep; k++) 
                // {
                //     for (int l = k; l >= sp + 1; l--) 
                //     {
                //         if (cloudCurvature[cloudSmoothness[l].ind] < cloudCurvature[cloudSmoothness[l-1].ind]) 
                //         {
                //             int temp = cloudSmoothness[l - 1].ind;
                //             cloudSmoothness[l - 1].ind = cloudSmoothness[l].ind;
                //             cloudSmoothness[l].ind = temp;
                //         }
                //     }
                // }

                int largestPickedNum = 0;

                //Xác nhận Corner point là các điểm có độ cong tương đối lớn cho từng vùng xét
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    //Kiểm tra điểm này không phải điểm bị loại ra và độ cong lớn hơn giá trị xác định nó là corner
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)  //edgeTheshold trong file param
                    {
                        largestPickedNum++;
                        //Chọn 20 điểm có độ cong lớn nhất vào tập hợp điểm chọn là corner
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;  //Đánh nhãn là độ cong lớn
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; //Định dấu cờ là các điểm corner và điểm loại ra được đánh dấu
                        //Lọc ra 5 điểm liên tiếp có khoảng cách tương đối gần trước và sau điểm có độ cong tương đối lớn
                        // Để ngăn sự phân nhóm của các điểm đối tượng, để các điểm đối tượng được phân bố đều theo các tính chất
                        for (int l = 1; l <= 5; l++)
                        {
                            //Khoảng cách theo tung toa do rieng biet VD: Velodyne là 1800
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)     //Nếu khoảng cách lớn hơn 10 pixel range image thì kết thúc vòng lặp không cần kiểm tra điểm kế tiếp
                                break;
                            cloudNeighborPicked[ind + l] = 1; //Đánh dấu cờ là điểm đã lọc
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)      //Nếu khoảng cách lớn hơn 10 pixel range image thì kết thúc vòng lặp không cần kiểm tra điểm kế tiếp
                                break;
                            cloudNeighborPicked[ind + l] = 1;  //Đánh dấu cờ là điểm đã lọc
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                          //Kiểm tra điểm này không phải điểm bị loại ra và độ cong lớn hơn giá trị xác định nó là plane
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;     //Đánh nhãn là độ cong nhỏ
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)      //Nếu khoảng cách lớn hơn 10 pixel range image thì kết thúc vòng lặp không cần kiểm tra điểm kế tiếp
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)     //Nếu khoảng cách lớn hơn 10 pixel range image thì kết thúc vòng lặp không cần kiểm tra điểm kế tiếp
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                //Các điểm có độ cong nhỏ gọi là mặt phẳng
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0 && cloudCurvature[k] < surfThreshold){       //Sửa ngày 18/11/2021      
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        //Publish cac doi tuong lien quan den corner va surface duoc trich xuat
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapping_pcd");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Buoc 2.\033[0m");

    ros::spin();

    return 0;
}
