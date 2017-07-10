#include "myslam.h"

//引入数学函数库
using namespace myslam;


using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

const double Pi=atanf(1)*4;
double pixel_size=0.00465;//像素尺寸4.65微米

char str_left[500];char str_right[500];char str_right2[500];char str_left2[500];//N是定义的常数，目的是为了读取足够长的行
MatrixXd p0(3,4);MatrixXd p1(3,4);//用来存储cali矩阵
Mat camera_mat(3, 3, CV_32FC1);//PNP方法中摄像机内参矩阵，p0中前3*3
Mat distCoeffs(4, 1, CV_32FC1);//畸变向量，因为数据已校正，故假设无畸变
MatrixXd pose_left(5000,6);MatrixXd pose_right(5000,6);//最终存储轨迹的矩阵
VectorXd pose1(6);VectorXd pose2(6);VectorXd pose3(6);VectorXd pose4(6);//对应i时刻和i+1时刻四影像外方，最终会存储到pose_left和pose_right里面
MatrixXd rotation(3,3);//旋转矩阵
double f;double tr;//焦距和左右相机距离


int minHessian = 8000;//surf方法参数
Mat descriptors_1,descriptors_2,descriptors_3,descriptors_4;//对应i时刻和i+1时刻四影像描述子
Mat descriptors_11,descriptors_22,descriptors_33,descriptors_44;//对应i时刻和i+1时刻四影像选用的特征点的描述子
vector< DMatch > match12,match13;
vector<KeyPoint> keypoints_1,keypoints_2,keypoints_3,keypoints_4;//对应i时刻和i+1时刻四影像特征点
vector<KeyPoint> keypoints_11,keypoints_22,keypoints_33,keypoints_44;//对应i时刻和i+1时刻四影像选用的特征点
vector<Point2f> p_left_keypoint1,p_right_keypoint1,p_left_keypoint2,p_right_keypoint2;//对应i时刻和i+1时刻四影像特征点的影像坐标
MatrixXd KeypointPose1,KeypointPose2,KeypointPose3,KeypointPose4;//存储特征点坐标和对应三维坐标,矩阵大小用的时候确定

//读取calib.txt的摄像机参数

struct BackCrossResidual
{
	/*
	* X, Y, Z, x, y 分别为观测值，f为焦距
	*/
	BackCrossResidual(double X, double Y, double Z, double x, double y, double f)
		:_X(X), _Y(Y), _Z(Z), _x(x), _y(y), _f(f){}

	/*
	* pBackCrossParameters：-2分别为Xs、Ys、Zs,3-5分别为Phi、Omega、Kappa
	*/
	template <typename T>
	bool operator () (const T * const pBackCrossParameters, T* residual) const
	{
		T dXs = pBackCrossParameters[0];
		T dYs = pBackCrossParameters[1];
		T dZs = pBackCrossParameters[2];
		T dPhi = pBackCrossParameters[3];
		T dOmega = pBackCrossParameters[4];
		T dKappa = pBackCrossParameters[5];

		T a1  = cos(dPhi)*cos(dKappa) - sin(dPhi)*sin(dOmega)*sin(dKappa);
		T a2  = -cos(dPhi)*sin(dKappa) - sin(dPhi)*sin(dOmega)*cos(dKappa);
		T a3  = -sin(dPhi)*cos(dOmega);
		T b1 = cos(dOmega)*sin(dKappa);
		T b2 = cos(dOmega)*cos(dKappa);
		T b3 = -sin(dOmega);
		T c1 = sin(dPhi)*cos(dKappa) + cos(dPhi)*sin(dOmega)*sin(dKappa);
		T c2 = -sin(dPhi)*sin(dKappa) + cos(dPhi)*sin(dOmega)*cos(dKappa);
		T c3 = cos(dPhi)*cos(dOmega);

		// 有两个残差
		residual[0]= T(_x) +T(_f) * T( (a1*(_X-dXs) + b1*(_Y-dYs) + c1*(_Z-dZs)) / ((a3*(_X-dXs) + b3*(_Y-dYs) + c3*(_Z-dZs))));
		residual[1]= T(_y) +T(_f) * T( (a2*(_X-dXs) + b2*(_Y-dYs) + c2*(_Z-dZs)) / ((a3*(_X-dXs) + b3*(_Y-dYs) + c3*(_Z-dZs))));

		return true;
	}

private:
	const double _X;
	const double _Y;
	const double _Z;
	const double _x;
	const double _y;
	const double _f;
};


void Matching(Mat img_left,Mat img_right,vector<DMatch> &good_matches,vector<KeyPoint> &keypoints_1,vector<KeyPoint> &keypoints_2)//用vector <string>做存贮文件名的容器
{
	ORB orb(200);
	orb(img_left, Mat(), keypoints_1, descriptors_1);
	orb(img_right, Mat(), keypoints_2, descriptors_2);

	BFMatcher matcher(NORM_L2);
	//FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	double mean_dis=0;
		for (int i=0; i<matches.size(); i++)
		{
		mean_dis=mean_dis+matches[i].distance;
		}
		mean_dis=mean_dis/matches.size();
	//RANSAC方法
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	Point2f pt;
	for (int i=0; i<ptCount; i++)
	{
		pt = keypoints_1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints_2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	Mat m_Fundamental;
	// 上面这个变量是基本矩阵
	vector<uchar> m_RANSACStatus;
	// 上面这个变量已经定义过，用于存储RANSAC后每个点的状态
	//cvFindFundamentalMat( const CvMat* points1, const CvMat* points2, CvMat* fundamental_matrix, int method=CV_FM_RANSAC, double param1=1., double param2=0.99, CvMat* status=NULL);
	//CV_FM_7POINT – 7-点算法，点数目= 7
	//CV_FM_8POINT – 8-点算法，点数目 >= 8
	//CV_FM_RANSAC – RANSAC 算法，点数目 >= 8
	//CV_FM_LMEDS - LMedS 算法，点数目 >= 8
	m_Fundamental = findFundamentalMat(p1,p2,m_RANSACStatus,0.1,0.99,CV_FM_RANSAC);
	//没有找到矩阵，返回0，找到一个矩阵返回1，多个矩阵返回3
	// 计算野点个数
	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	for( int i = 0; i <matches.size(); i++)
	{ 
		if(m_RANSACStatus[i] == 1 && matches[i].distance<=mean_dis)//&& matches[i].distance<mean_dis
		{ 
			good_matches.push_back( matches[i]);
			keypoints_11.push_back(keypoints_1[matches[i].queryIdx]);
			descriptors_11.push_back(descriptors_1.row(matches[i].queryIdx));
		}
	}
	//cout<<"goodmatches数为"<<"\t"<<good_matches.size()<<endl;
	//Mat img_matches;
	//drawMatches( img_left, keypoints_1,img_right, keypoints_2, good_matches, img_matches );
	//IplImage ipl_img(img_matches); 
	//cvNamedWindow("Matches",0);
	//cvShowImage("Matches", &ipl_img);
	//cvWaitKey(0);

}

void ExactPixel(vector<DMatch> matches,vector<KeyPoint> keypoints_left,vector<KeyPoint> keypoints_right,vector<Point2f> &p_left_point,vector<Point2f> &p_right_point)
{
	for(unsigned int i=0;i<matches.size();i++)
	{
		p_left_point.push_back(keypoints_left[ matches[i].queryIdx ].pt);
		p_right_point.push_back(keypoints_right[ matches[i].trainIdx ].pt);
	}
}

void Rotation(double pitch,double yaw,double roll,MatrixXd &rotation)
{
	MatrixXd m1(3,3);MatrixXd m2(3,3);MatrixXd m3(3,3);
	m1(0,0)=cos(pitch);m1(0,1)=0;m1(0,2)=-sin(pitch);m1(1,0)=0;m1(1,1)=1;m1(1,2)=0;m1(2,0)=sin(pitch);m1(2,1)=0;m1(2,2)=cos(pitch);
	m2(0,0)=1;m2(0,1)=0;m2(0,2)=0;m2(1,0)=0;m2(1,1)=cos(yaw);m2(1,2)=-sin(yaw);m2(2,0)=0;m2(2,1)=sin(yaw);m2(2,2)=cos(yaw);
	m3(0,0)=cos(roll);m3(0,1)=-sin(roll);m3(0,2)=0;m3(1,0)=sin(roll);m3(1,1)=cos(roll);m3(1,2)=0;m3(2,0)=0;m3(2,1)=0;m3(2,2)=1;
	rotation=m2*m1*m3;
}

void Intersection(MatrixXd cali_left,MatrixXd cali_right,VectorXd pose_left,VectorXd pose_right,vector<Point2f> p_left_keypoint,vector<Point2f> p_right_keypoint,MatrixXd &KeypointPose)
{
	double cx1=cali_left(0,2);
	double cy1=cali_left(1,2);
	double cx2=cali_right(0,2);
	double cy2=cali_right(1,2);

	double Xs1=pose_left(0);double Xs2=pose_right(0);
	double Ys1=pose_left(1);double Ys2=pose_right(1);
	double Zs1=pose_left(2);double Zs2=pose_right(2);

	double pitch=pose_left(3);double yaw=pose_left(4);double roll=pose_left(5);//本项目中Rotation1,Rotation2相等

	Mat rvec(3, 1, CV_64FC1);
	Mat d(3, 3, CV_64FC1);
	rvec.at<double>(0)= pitch;rvec.at<double>(1)= yaw;rvec.at<double>(2)= roll;
	Rodrigues(rvec, d);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			rotation(i, j)=d.at<double>(i, j);
	double a1=rotation(0,0);double a2=rotation(0,1);double a3=rotation(0,2);double b1=rotation(1,0);double b2=rotation(1,1);double b3=rotation(1,2);double c1=rotation(2,0);double c2=rotation(2,1);double c3=rotation(2,2);
	for(unsigned int i=0;i<p_left_keypoint.size();i++)
	{
		MatrixXd L0(4,3);VectorXd L1(4);MatrixXd X;
		L0(0, 0) = -f*a1 + a3*(p_left_keypoint[i].x - cx1)*pixel_size; L0(0, 1) = -f*b1 + b3*(p_left_keypoint[i].x - cx1)*pixel_size; L0(0, 2) = -f*c1 + c3*(p_left_keypoint[i].x - cx1)*pixel_size;
		L0(1, 0) = -f*a2 + a3*(p_left_keypoint[i].y - cy1)*pixel_size; L0(1, 1) = -f*b2 + b3*(p_left_keypoint[i].y - cy1)*pixel_size; L0(1, 2) = -f*c2 + c3*(p_left_keypoint[i].y - cy1)*pixel_size;
		L0(2, 0) = -f*a1 + a3*(p_right_keypoint[i].x - cx2)*pixel_size; L0(2, 1) = -f*b1 + b3*(p_right_keypoint[i].x - cx2)*pixel_size; L0(2, 2) = -f*c1 + c3*(p_right_keypoint[i].x - cx2)*pixel_size;
		L0(3, 0) = -f*a2 + a3*(p_right_keypoint[i].y - cy2)*pixel_size; L0(3, 1) = -f*b2 + b3*(p_right_keypoint[i].y - cy2)*pixel_size; L0(3, 2) = -f*c2 + c3*(p_right_keypoint[i].y - cy2)*pixel_size;

		L1(0) = -(f*a1*Xs1 + f*b1*Ys1 + f*c1*Zs1) + a3*Xs1*(p_left_keypoint[i].x - cx1)*pixel_size + b3*Ys1*(p_left_keypoint[i].x - cx1)*pixel_size + c3*Zs1*(p_left_keypoint[i].x - cx1)*pixel_size;
		L1(1) = -(f*a2*Xs1 + f*b2*Ys1 + f*c2*Zs1) + a3*Xs1*(p_left_keypoint[i].y - cy1)*pixel_size + b3*Ys1*(p_left_keypoint[i].y - cy1)*pixel_size + c3*Zs1*(p_left_keypoint[i].y - cy1)*pixel_size;
		L1(2) = -(f*a1*Xs2 + f*b1*Ys2 + f*c1*Zs2) + a3*Xs2*(p_right_keypoint[i].x - cx2)*pixel_size + b3*Ys2*(p_right_keypoint[i].x - cx2)*pixel_size + c3*Zs2*(p_right_keypoint[i].x - cx2)*pixel_size;
		L1(3) = -(f*a2*Xs2 + f*b2*Ys2 + f*c2*Zs2) + a3*Xs2*(p_right_keypoint[i].y - cy2)*pixel_size + b3*Ys2*(p_right_keypoint[i].y - cy2)*pixel_size + c3*Zs2*(p_right_keypoint[i].y - cy2)*pixel_size;
		X = L0.jacobiSvd(ComputeThinU | ComputeThinV).solve(L1);
		KeypointPose(i,0)=X(0);KeypointPose(i,1)=X(1);KeypointPose(i,2)=X(2);
	}

}

void MatchNext(Mat img_1,Mat img_3,vector<KeyPoint> keypoints_1,MatrixXd KeypointPose1,MatrixXd &KeypointPose3)//输入前一时刻左影像，后一时刻影像,前一时刻的匹配结果，输出前一时刻左影像和后一时刻的匹配结果及其三维坐标
{
	ORB orb(200);
	orb(img_3, Mat(), keypoints_3, descriptors_3);
	//orb(img_1, Mat(), keypoints_11, descriptors_11);

	BFMatcher matcher(NORM_L2);
	vector< DMatch > matches;
	matcher.match( descriptors_11, descriptors_3, matches );

		double mean_dis=0;
		for (int i=0; i<matches.size(); i++)
		{
		mean_dis=mean_dis+matches[i].distance;

		}
		mean_dis=mean_dis/matches.size();

	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	Point2f pt;
	for (int i=0; i<ptCount; i++)
	{
		pt = keypoints_11[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints_3[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	Mat m_Fundamental;
	// 上面这个变量是基本矩阵
	vector<uchar> m_RANSACStatus;
	m_Fundamental = findFundamentalMat(p1, p2,m_RANSACStatus,1,0.9,CV_FM_RANSAC);
	int p_count_need=400;
	for( int i = 0; i <min(ptCount,p_count_need); i++ )
	{ 
		if(m_RANSACStatus[i] == 1 && matches[i].distance<=mean_dis)//&&matches[i].distance<=mean_dis 
		{ 
			match13.push_back( matches[i]);
		}
	}
	cout<<"match13数为"<<"\t"<<match13.size()<<endl;
	
	
	//Mat img_matches;
	//drawMatches( img_1, keypoints_11,img_3, keypoints_3, match13, img_matches );
	//IplImage ipl_img(img_matches); 
	//cvNamedWindow("Match_Next",0);
	//cvShowImage("Match_Next", &ipl_img);
	//cvWaitKey(0);
	p_left_keypoint1.clear();//清除以保存数据

	ExactPixel(match13,keypoints_11,keypoints_3,p_left_keypoint1,p_left_keypoint2);

	int k;
	KeypointPose3.resize(match13.size(),5);
	for(unsigned int i=0;i<match13.size();i++)
	{
		k=match13[i].queryIdx;
		KeypointPose3(i,0)=p_left_keypoint2[i].x;
		KeypointPose3(i,1)=p_left_keypoint2[i].y;
		KeypointPose3(i,2)=KeypointPose1(k,2);
		KeypointPose3(i,3)=KeypointPose1(k,3);
		KeypointPose3(i,4)=KeypointPose1(k,4);
	}
	//cout<<KeypointPose3<<endl;

}

void Resection(MatrixXd cali_left,MatrixXd KeypointPose3,VectorXd &pose3)
{
	
	double cx1=cali_left(0,2);
	double cy1=cali_left(1,2);

	int nCount = KeypointPose3.rows();
	MatrixXd Keypoint3(nCount,2);
	double dBackCrossParameters[6] = {0};   //初值

	for(int i=0;i<nCount;i++)
	{
		dBackCrossParameters[0]+= KeypointPose3(i,2); //X
		dBackCrossParameters[1]+= KeypointPose3(i,3); //Y
		Keypoint3(i,0)=(KeypointPose3(i,0)-cx1)*pixel_size;
		Keypoint3(i,1)=-(KeypointPose3(i,1)-cy1)*pixel_size;
	}
	//cout<<KeypointPose3<<endl;
	//cout<<Keypoint3<<endl;
	dBackCrossParameters[0]/= nCount;
	dBackCrossParameters[1]/= nCount;
	dBackCrossParameters[2]= 100*f;

	Problem problem; 
	for (int i = 0; i < nCount;i++)
	{
		BackCrossResidual *pResidualX = new BackCrossResidual(KeypointPose3(i,2),KeypointPose3(i,3), KeypointPose3(i,4),Keypoint3(i,0), Keypoint3(i,1),f);
		problem.AddResidualBlock(new AutoDiffCostFunction<BackCrossResidual, 2, 6>(pResidualX), NULL,dBackCrossParameters); 
	}

	Solver::Options m_options;
	Solver::Summary m_summary;
	m_options.max_num_iterations = 30;
	m_options.linear_solver_type = ceres::DENSE_QR;
	// m_options.minimizer_progress_to_stdout = true;

	Solve(m_options, &problem,&m_summary);

	//fprintf(stdout,"Xs=%lfYs=%lf Zs=%lf\n",dBackCrossParameters[0],dBackCrossParameters[1],dBackCrossParameters[2]);
	//fprintf(stdout,"Phi=%lfOmega=%lf Kappa=%lf\n",dBackCrossParameters[3],dBackCrossParameters[4],dBackCrossParameters[5]);
	//fprintf(stdout,"%s\n",m_summary.FullReport().c_str());
	pose3(0)=dBackCrossParameters[0];pose3(1)=dBackCrossParameters[1];
	pose3(2)=dBackCrossParameters[2];pose3(3)=dBackCrossParameters[3];
	pose3(4)=dBackCrossParameters[4];pose3(5)=dBackCrossParameters[5];
}



int main(int argc, char** argv)
{

	ifstream infile_left("C://Users/Zero/desktop/image_0/imagelist_left.txt");// imagelist.txt文件内容是生成的文件夹中所有图像的路径,每个图像的路径为一行
	ifstream infile_right("C://Users/Zero/desktop/image_1/imagelist_right.txt");
	int n=0;//用来表示说读/写文本的行数
	FILE *fp1;//结果文件
	fp1 = fopen("C:\\Users\\Zero\\Desktop\\vo_orb_epnp_ba.txt", "a+");
	p0<< 718.856,0,607.1928,0,
		0,718.856,185.2157,0,
		0,0,1,0;
	p1<<718.856,0,607.1928,-386.1448,
		0,718.856,185.2157,0,
		0,0,1,0;
	f=p0(0,0)*pixel_size;
	tr=-p1(0,3)/f*pixel_size*1000;
	
	pose_left(0,0)=0;pose_left(0,1)=0;pose_left(0,2)=0;pose_left(0,3)=0;pose_left(0,4)=0;pose_left(0,5)=0;
	pose_right(0,0)=tr;pose_right(0,1)=0;pose_right(0,2)=0;pose_right(0,3)=0;pose_right(0,4)=0;pose_right(0,5)=0;
	double pitch1=pose_left(0,3);double yaw1=pose_left(0,4);double roll1=pose_left(0,5);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			camera_mat.at<float>(i, j) = p0(i, j);
		}
	}

	distCoeffs.at<float>(0) = 0;distCoeffs.at<float>(1) = 0;
	distCoeffs.at<float>(2) = 0;distCoeffs.at<float>(3) = 0;

	infile_left.getline(str_left, sizeof(str_left));
	Mat img_left=imread(str_left,0 );
	infile_right.getline(str_right, sizeof(str_right));
	Mat img_right=imread(str_right,0);//读取n时刻影像
	

	while(n<4540)//!infile_left.eof()
	{
		DWORD start_time=GetTickCount();
		infile_left.getline(str_left2, sizeof(str_left2));
		infile_right.getline(str_right2, sizeof(str_right2));//读取n+1时刻的两幅影像地址
		Mat img_left2=imread(str_left2,0 );
		Mat img_right2=imread(str_right2,0);//读取n+1时刻的两幅影像

		Matching(img_left,img_right,match12,keypoints_1,keypoints_2);//输入左右影像输出matches和keypoints

		ExactPixel(match12,keypoints_1,keypoints_2,p_left_keypoint1,p_right_keypoint1);//输入matches左右匹配坐标向量，输出匹配点坐标

		for(int i=0;i<=5;i++)
		{
			pose1(i) = 0; pose2(i) = 0;
		}
		pose2(0) = tr;

		int j=match12.size();//12匹配点数
		MatrixXd KeypointPose(j,3);//存储三维坐标
		Intersection(p0,p1,pose1,pose2,p_left_keypoint1,p_right_keypoint1,KeypointPose);

		KeypointPose1.resize(j,5);
		for( int i=0;i<j;i++)
		{
			KeypointPose1(i,0)=p_left_keypoint1[i].x;
			KeypointPose1(i,1)=p_left_keypoint1[i].y;
			KeypointPose1(i,2)=KeypointPose(i,0);
			KeypointPose1(i,3)=KeypointPose(i,1);
			KeypointPose1(i,4)=KeypointPose(i,2);

		}

		//特征点编号、观测次数、观测到的帧及其对应的像素坐标、三维坐标存入mappoints



		MatchNext(img_left,img_left2,keypoints_1,KeypointPose1,KeypointPose3);//输入i和i+1时刻左影像与i时刻左影像特征点及其地面坐标，输出i+1时刻左影像匹配特征点及其地面坐标

		//  solvePnPRansac方法求i+1时刻左相机姿态
		Mat rvec(3, 1, CV_64FC1);
		Mat r(3, 1, CV_64FC1),r_before(3, 1, CV_64FC1); //总旋转向量
		Mat dr(3, 3, CV_64FC1),d_before(3, 3, CV_64FC1);//总旋转阵
		Mat tvec(3, 1, CV_64FC1);
		Mat t(3, 1, CV_64FC1);
		Mat d(3, 3, CV_64FC1);
		
		vector<Point3f> list_points3d_model_match(KeypointPose3.rows());
		vector<Point2f> list_points2d_scene_match(KeypointPose3.rows());
		for (int i = 0; i < KeypointPose3.rows(); i++)
		{
			list_points2d_scene_match[i].x = KeypointPose3(i, 0);
			list_points2d_scene_match[i].y = KeypointPose3(i, 1);
			list_points3d_model_match[i].x = KeypointPose3(i, 2);
			list_points3d_model_match[i].y = KeypointPose3(i, 3);
			list_points3d_model_match[i].z = KeypointPose3(i, 4);
		}

		solvePnPRansac(list_points3d_model_match, list_points2d_scene_match, camera_mat, distCoeffs, rvec, tvec, false,100,3.0f,100,noArray(), CV_EPNP);
		Rodrigues(rvec,d);
		d = d.inv();

		Rodrigues(d, rvec);

		for (int i = 0; i<3; i++)
		{
			r_before.at<double>(i) = pose_left(n, i+3);
		}
		Rodrigues(r_before, d_before);
		dr = d_before*d;
		Rodrigues(dr,r);

		t = dr*tvec;
		for (int i = 0; i<3; i++)
		{
			pose3(i)=-t.at<double>(i);
		}

		pose3(3) =rvec.at<double>(0); pose3(4) = rvec.at<double>(1); pose3(5) =rvec.at<double>(2);


		for(int i=0;i<3;i++)
		{
			pose_left(n+1,i)=pose_left(n,i)+pose3(i);
		}
		pose_left(n+1,3)=r.at<double>(0);pose_left(n+1,4)=r.at<double>(1);pose_left(n+1,5)=r.at<double>(2);
		cout<<pose_left(n+1,0)<<"     "<<pose_left(n+1,1)<<"     "<<pose_left(n+1,2)<<"     "<<pose_left(n+1,3)<<"     "<<pose_left(n+1,4)<<"     "<<pose_left(n+1,5)<<endl;

		fprintf(fp1,"%lf %1f %1f %lf %1f %1f\n",pose_left(n+1,0),pose_left(n+1,1),pose_left(n+1,2),pose_left(n+1,3),pose_left(n+1,4),pose_left(n+1,5));


		img_left=img_left2.clone();//i+1图像变为i时刻
		img_right=img_right2.clone();
		
		
		img_left2.release();img_right2.release();
		match12.clear();match13.clear();
		p_left_keypoint1.clear();p_right_keypoint1.clear();p_left_keypoint2.clear();p_right_keypoint2.clear();
		keypoints_1.clear(); keypoints_2.clear(); keypoints_11.clear();//对应i时刻和i+1时刻四影像选用的特征点
		descriptors_1.release();descriptors_2.release();descriptors_11.release();
		
		list_points3d_model_match.clear(); list_points2d_scene_match.clear();
		DWORD end_time=GetTickCount();
		DWORD Subtime = (end_time-start_time);
		cout<<"这是第"<<n+1<<"时刻的位置"<<endl;
		cout<<"计算时间为:"<<Subtime<<endl;
		n++;


	}
	fclose(fp1);
	infile_left.close();
	infile_right.close();
	cout<<n<<endl;
	system("pause");
	return 0;
}
