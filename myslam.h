pragma once

#include "targetver.h"
#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include<vector>
#include<list>
#include <fstream>
#include<cstdlib> 
#include <iostream>
#include <string>
#include <ctype.h>


#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "cvaux.hpp"
#include <opencv2\opencv.hpp>
#include "opencv.hpp"
#include "imgproc/imgproc.hpp"
#include "cvaux.h"
#include "nonfree/features2d.hpp"
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>


//引入数学函数库
#include<math.h> 
#include<cmath>
#include<cstring>
#include<cfloat>

#include <Eigen/Dense>//矩阵库
#include <Eigen/Cholesky>

#include<glog\logging.h>
#include <ceres\ceres.h>
#include <gflags\gflags.h>

#include "sophus\se3.hpp"
#include "sophus\so3.hpp"
#include "sophus\sophus.hpp"
#include "sophus\sim3.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Sophus;
