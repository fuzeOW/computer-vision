/*creater: Lingyu Wang*/
#include <caffe/caffe.hpp>
#include <caffe/proto/caffe.pb.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>//ly add
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/cudafeatures2d.hpp"	//ly add 20171218
#include "opencv2/opencv_modules.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/cudabgsegm.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/features2d.hpp"

//#include "opencv2/xfeatures2d/cuda.hpp"
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;
//scaling process: input pictures are images of any size.
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


double camD[9] = { 713.42499, 0, 307.28579,
		   0, 714.95450, 184.69909,
		   0, 0, 1 };
Mat camera_matrix = Mat(3, 3, CV_64FC1, camD);
double distCoeffD[5] = { -0.17964, 0.20056, -0.00512, -0.00247, 0.00000 };
Mat distortion_coefficients = Mat(5, 1, CV_64FC1, distCoeffD);

vector<Point3f> objP;
Mat objPM;
vector<double> rv(3), tv(3);
Mat rvec(rv), tvec(tv);
double rm[9];
Mat rotM(3, 3, CV_64FC1, rm);

vector<Point2f> points;
Mat image, frame;
int framenum = 0;
int i_frame = 0;

int TemplateWidth;
int TemplateHeight;

Mat img1 = imread("/home/nvidia/project_wly_gpu/logo.jpg",CV_LOAD_IMAGE_GRAYSCALE);  //ly add
//Mat img1 = imread("/home/nvidia/project_wly_gpu/logo.jpg");
GpuMat img1GPU(img1);
//std::vector<cv::KeyPoint> img1_keypointsGPU;
cv::cuda::GpuMat img1_keypointsGPU;
cv::cuda::GpuMat img1_descriptorsGPU;

Mat img2;
GpuMat img2GPU(img2);
//std::vector<cv::KeyPoint> img2_keypointsGPU;
cv::cuda::GpuMat img2_keypointsGPU;
cv::cuda::GpuMat img2_descriptorsGPU;

//cuda::cvtColor(img1GPU, img1GPU, COLOR_BGR2GRAY);
//cuda::cvtColor(img2GPU, img2GPU, COLOR_BGR2GRAY);

vector<KeyPoint> keypoints1, keypoints2;
cv::Mat descriptors1, descriptors2;

cv::VideoCapture capture(1);

bool init(GpuMat img1GPU)
{
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 256);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 256);


	if (!capture.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return false;
	}
	else
	{
		cout << "Capturing...\n";

		TemplateWidth = img1GPU.cols;
		TemplateHeight = img1GPU.rows;
		std::cout << "TemplateWidth = " << TemplateWidth << endl;
		std::cout << "TemplateHeight = " << TemplateHeight << endl;

		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice()); 	//ly add 20171219

                //cuda::cvtColor(img1GPU, img1GPU, CV_BGR2GRAY);
		cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(30000,1.2f,8,31,0,2,31,20,false);	//ly add 20171222
                //Ptr<cuda::ORB> d_orb = cuda::ORB::create(500, 1.2f, 6, 31, 0, 2, 0, 31, 20,true);


                orb->detectAndComputeAsync(img1GPU, cuda::GpuMat(), img1_keypointsGPU, img1_descriptorsGPU);

    		std::cout << "img1_keypointsGPU=" << img1_keypointsGPU.size() << std::endl;

		//imshow("logo",img1);
		//cvWaitKey(30);

		// matching descriptors
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
		vector<DMatch> matches;
		matcher->match(img1_descriptorsGPU, img2_descriptorsGPU, matches);
		cout << "find out total " << matches.size() << " matches" << endl;

		// downloading results
		orb -> convert(img1_keypointsGPU, keypoints1);
		orb -> convert(img2_keypointsGPU, keypoints2);

		img1_descriptorsGPU.download(descriptors1);
		img2_descriptorsGPU.download(descriptors2);

		cout << "FOUND " << img1_keypointsGPU.size() << " img1_keypoints on first image" << endl;
		cvWaitKey(0);
	}
	//namedWindow("capture", 1);

	objP.push_back(Point3f(54, 0, 0));
	objP.push_back(Point3f(54, 54, 0));
	objP.push_back(Point3f(0, 54, 0));
	objP.push_back(Point3f(0, 0, 0));
	Mat(objP).convertTo(objPM, CV_32F);

	return true;
}

class RobustMatcher
{
private:
	Ptr<FeatureDetector>detector;
	Ptr<DescriptorExtractor>extractor;
	Ptr<cv::cuda::DescriptorMatcher> matcher;

	float ratio;
	bool refineF;
	double distance;
	double confidence;


public:

	RobustMatcher() :ratio(0.95f), refineF(true), confidence(0.99), distance(3.0) {

		detector = cv::cuda::ORB::create();//ly add

		extractor = cv::ORB::create();//ly add

		//matcher = cv::BFMatcher::create();
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
	}

	void setFeatureDetector(Ptr<FeatureDetector>& detect)
	{
		detector = detect;
	}

	void setDescriptorExtractor(Ptr<DescriptorExtractor>& desc)
	{
		extractor = desc;
	}

	void setDescriptorMatcher(Ptr<cv::cuda::DescriptorMatcher>& match)
	{
		matcher = match;
	}

	void setConfidenceLevel(double conf)
	{
		confidence = conf;
	}

	void setMinDistanceToEpipolar(double dist)
	{
		distance = dist;
	}

	void setRadio(float rat)
	{
		ratio = rat;
	}

	Mat match(GpuMat& image1, GpuMat& image2, vector<DMatch>& matches, vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2);

	Mat ransacTest(const vector<DMatch>& matches, const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, vector<DMatch>& outMatches);

	void symmetryTest(const vector<vector<DMatch> >& matches1, const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches);

	int ratioTest(vector<vector<DMatch> >& matches);

};

int RobustMatcher::ratioTest(std::vector <std::vector<cv::DMatch> > & matches)
{
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<DMatch> >::iterator
		matchIterator = matches.begin();
		matchIterator != matches.end(); ++matchIterator)
	{
		// if 2 NN has been identified
		if (matchIterator->size() > 1)
		{
			// check distance ratio
			if ((*matchIterator)[0].distance /
				(*matchIterator)[1].distance > ratio)
			{
				matchIterator->clear(); // remove match
				removed++;
			}
		}
		else
		{ // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
}

// Insert symmetrical matches in symMatches vector
void RobustMatcher::symmetryTest(
	const vector<vector<DMatch> >& matches1,
	const vector<vector<DMatch> >& matches2,
	vector<DMatch>& symMatches)
{
	// for all matches image 1 -> image 2
	for (vector<vector<DMatch> >::
		const_iterator matchIterator1 = matches1.begin();
		matchIterator1 != matches1.end(); ++matchIterator1)
	{
		// ignore deleted matches
		if (matchIterator1->size() < 2)
			continue;
		// for all matches image 2 -> image 1
		for (vector<vector<DMatch> >::
			const_iterator matchIterator2 = matches2.begin();
			matchIterator2 != matches2.end(); ++matchIterator2)
		{
			// ignore deleted matches
			if (matchIterator2->size() < 2)
				continue;
			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx ==
				(*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx ==
				(*matchIterator1)[0].trainIdx)
			{
				// add symmetrical match
				symMatches.push_back(
					DMatch((*matchIterator1)[0].queryIdx,
					(*matchIterator1)[0].trainIdx,
						(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

// Identify good matches using RANSAC
// Return fundemental matrix
cv::Mat RobustMatcher::ransacTest(
	const vector<DMatch>& matches,
	const std::vector<cv::KeyPoint>& keypoints1,
	const std::vector<cv::KeyPoint>& keypoints2,
	vector<DMatch>& outMatches)
{
	// Convert keypoints into Point2f
	vector<Point2f> points1, points2;
	cv::Mat fundemental;
	for (std::vector<cv::DMatch>::
		const_iterator it = matches.begin();
		it != matches.end(); ++it)
	{
		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	// Compute F matrix using RANSAC
	vector<uchar> inliers(points1.size(), 0);
	if (points1.size() > 0 && points2.size() > 0)
	{
		Mat fundemental = findFundamentalMat(
			Mat(points1), Mat(points2), // matching points
			inliers,       // match status (inlier or outlier)
			CV_FM_RANSAC, // RANSAC method
			distance,      // distance to epipolar line
			confidence); // confidence probability
						 // extract the surviving (inliers) matches
		vector<uchar>::const_iterator
			itIn = inliers.begin();
		vector<DMatch>::const_iterator
			itM = matches.begin();
		// for all matches
		for (; itIn != inliers.end(); ++itIn, ++itM)
		{
			if (*itIn)
			{ // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		if (refineF)
		{
			// The F matrix will be recomputed with
			// all accepted matches
			// Convert keypoints into Point2f
			// for final F computation
			points1.clear();
			points2.clear();
			for (vector<DMatch>::
				const_iterator it = outMatches.begin();
				it != outMatches.end(); ++it)
			{
				// Get the position of left keypoints
				float x = keypoints1[it->queryIdx].pt.x;
				float y = keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x, y));
				// Get the position of right keypoints
				x = keypoints2[it->trainIdx].pt.x;
				y = keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x, y));
			}
			// Compute 8-point F from all accepted matches
			if (points1.size() > 0 && points2.size() > 0)
			{
				fundemental = findFundamentalMat(
					Mat(points1), Mat(points2), // matches
					CV_FM_8POINT); // 8-point method
			}
		}
	}
	return fundemental;
}

void codeRotateByZ(double x, double y, double thetaz, double& outx, double& outy)
{
	double x1 = x;
	double y1 = y;
	double rz = thetaz * CV_PI / 180;
	outx = cos(rz) * x1 - sin(rz) * y1;
	outy = sin(rz) * x1 + cos(rz) * y1;
									   //cout << "outx = " << outx << endl;
}


void codeRotateByY(double x, double z, double thetay, double& outx, double& outz)
{
	double x1 = x;
	double z1 = z;
	double ry = thetay * CV_PI / 180;
	outx = cos(ry) * x1 + sin(ry) * z1;
	outz = cos(ry) * z1 - sin(ry) * x1;
	//cout << "outz = " << outz << endl;
}


void codeRotateByX(double y, double z, double thetax, double& outy, double& outz)
{
	double y1 = y;
	double z1 = z;
	double rx = thetax * CV_PI / 180;
	outy = cos(rx) * y1 - sin(rx) * z1;
	outz = cos(rx) * z1 + sin(rx) * y1;
}

void getPlanarSurface(vector<Point2f>& imgP) {

	Rodrigues(rotM, rvec);
	solvePnP(objPM, Mat(imgP), camera_matrix, distortion_coefficients, rvec, tvec);

	Rodrigues(rvec, rotM);

	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];


	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;

	ofstream  fout;
	fout.open("D:\\pnp_theta.txt", ios::app);
	fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	cout << "The three_axis rotation angle of the camera:" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	fout.close();

	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	double x = tx, y = ty, z = tz;

	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	double Cx = x*-1;
	double Cy = y*-1;
	double Cz = z*-1;

	ofstream  fout2;
	fout2.open("D:\\pnp_world.txt", ios::app);
	fout2 << Cx << endl << Cy << endl << Cz << endl;
	cout << "The world coordinates of the camera:" << Cx << ", " << Cy << ", " << Cz << endl;
	fout2.close();

	cout << "Three-directional displacement of the target: " << tv[0] << ", " << tv[1] << ", " << tv[2] << endl;

	ofstream  fout3;
	fout3.open("D:\\pnp_target.txt", ios::app);
	fout3 << tv[0] << ", " << tv[1] << ", " << tv[2] << endl;
	fout3.close();
}
// Match feature points using symmetry test and RANSAC
// returns fundemental matrix
Mat RobustMatcher::match(GpuMat& image1,
	GpuMat& image2,
	vector<DMatch>& matches,
	vector<cv::KeyPoint>& keypoints1,
	vector<cv::KeyPoint>& keypoints2)
{
	// 1a. Detection of the ORB features
	double t1 = getTickCount();
	detector->detect(image1, keypoints1);
	detector->detect(image2, keypoints2);
	double t2 = getTickCount() - t1;
	t2 = 1000 * t2 / getTickFrequency();
	cout << "detect time " << t2 << " ms " << endl;

	Mat fundemental;

	// 1b. Extraction of the ORB descriptors
	t1 = getTickCount();
	cv::Mat descriptors1, descriptors2;
	extractor->compute(image1, keypoints1, descriptors1);
	extractor->compute(image2, keypoints2, descriptors2);

	cout << "Numbers of keypoints of Image1 : " << keypoints1.size() << endl;
	cout << "Numbers of keypoints of Image2 : " << keypoints2.size() << endl;

	t2 = getTickCount() - t1;
	t2 = 1000 * t2 / getTickFrequency();
	cout << "extractor time " << t2 << " ms " << endl;

	// 2. Match the two image descriptors
	// Construction of the matcher
	//cv::BruteForceMatcher<cv::L2<float>> matcher;
	// from image 1 to image 2
	// based on k nearest neighbours (with k=2)
	if ((descriptors2.rows > 0) && (descriptors2.cols > 0)) {

		vector<vector<DMatch> > matches1;
		matcher->knnMatch(descriptors1, descriptors2,
			matches1, // vector of matches (up to 2 per entry)
			2);        // return 2 nearest neighbours
					   // from image 2 to image 1
					   // based on k nearest neighbours (with k=2)
		vector<vector<DMatch> > matches2;
		matcher->knnMatch(descriptors2, descriptors1,
			matches2, // vector of matches (up to 2 per entry)
			2);        // return 2 nearest neighbours

					   // 3. Remove matches for which NN ratio is
					   // > than threshold
					   // clean image 1 -> image 2 matches
		int removed = ratioTest(matches1);
		// clean image 2 -> image 1 matches
		removed = ratioTest(matches2);

		// 4. Remove non-symmetrical matches
		vector<DMatch> symMatches;
		symmetryTest(matches1, matches2, symMatches);

		//=========================================²âÊÔŽúÂë
		Mat img_matches;
		t1 = getTickCount();
		drawMatches(image1, keypoints1, image2, keypoints2,
			symMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		t2 = getTickCount() - t1;
		t2 = 1000 * t2 / getTickFrequency();
		cout << "match time = " << t2 << " ms " << endl;
		//cout << "Numbers of match  " << img_matches.size() << endl;
		/*imshow("Test",img_matches);*/
		//cvWaitKey(0);
		//=========================================²âÊÔŽúÂë

		// 5. Validate matches using RANSAC
		fundemental = ransacTest(symMatches,
			keypoints1, keypoints2, matches);

		//=========================================²âÊÔŽúÂë
		vector<Point2f> obj;
		vector<Point2f> scene;

		for (int i = 0; i < symMatches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints1[symMatches[i].queryIdx].pt);
			scene.push_back(keypoints2[symMatches[i].trainIdx].pt);
		}
		if (obj.size() >= 4 && scene.size() >= 4) {
			Mat H = findHomography(obj, scene, CV_RANSAC, 2);
			//-- Get the corners from the image_1 ( the object to be "detected" )
			vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0, 0);
			obj_corners[1] = cvPoint(image1.cols, 0);
			obj_corners[2] = cvPoint(image1.cols, image1.rows);
			obj_corners[3] = cvPoint(0, image1.rows);
			vector<Point2f> scene_corners(4);

			perspectiveTransform(obj_corners, scene_corners, H);

			/*for (int i = 0; i < 4; i++)
			{
			scene_corners[i].x += image1.cols;
			}*/

			cout << "scene_corners[0] = " << scene_corners[0] << endl;        //ly add
			cout << "scene_corners[1] = " << scene_corners[1] << endl;
			cout << "scene_corners[2] = " << scene_corners[2] << endl;
			cout << "scene_corners[3] = " << scene_corners[3] << endl;

			/*line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
			line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
			line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
			line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);*/

			//imshow("Test", img_matches);
			cout << "Numbers of match :" << symMatches.size() << endl;

			//ly add 20170506
			/*char file_name[256] = "match_result";
			IplImage *s = &IplImage(img_matches);
			char key = cvWaitKey(20);
			sprintf_s(file_name, "D:\\image_result\\%d%s", ++i_frame, ".jpg");
			cvSaveImage(file_name, s);*/
			//end

			/*line(img2, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
			line(img2, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
			line(img2, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
			line(img2, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);

			imshow("Test", img2);*/

			/*line(img2, scene_corners[0] - Point2f(img1.cols, 0), scene_corners[1] - Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
			line(img2, scene_corners[1] - Point2f(img1.cols, 0), scene_corners[2] - Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
			line(img2, scene_corners[2] - Point2f(img1.cols, 0), scene_corners[3] - Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
			line(img2, scene_corners[3] - Point2f(img1.cols, 0), scene_corners[0] - Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);*/
			//imshow("Object detection", img2);

			//cout << "(1):scene_corners[0] - Point2f(img1.cols, 0) = " << scene_corners[1] - Point2f(img1.cols, 0) << endl;
			//cout << "(2):scene_corners[1] - Point2f(img1.cols, 0) = " << scene_corners[2] - Point2f(img1.cols, 0) << endl;
			//cout << "(3):scene_corners[2] - Point2f(img1.cols, 0) = " << scene_corners[3] - Point2f(img1.cols, 0) << endl;
			//cout << "(4):scene_corners[3] - Point2f(img1.cols, 0) = " << scene_corners[0] - Point2f(img1.cols, 0) << endl;

			//points.push_back(scene_corners[0] - Point2f(img1.cols, 0));
			//points.push_back(scene_corners[1] - Point2f(img1.cols, 0));// 0-----1
			//points.push_back(scene_corners[2] - Point2f(img1.cols, 0));//
			//points.push_back(scene_corners[3] - Point2f(img1.cols, 0));// 3-----2

			points.push_back(cv::Point2f(scene_corners[0]));
			points.push_back(cv::Point2f(scene_corners[1]));// 0-----1
			points.push_back(cv::Point2f(scene_corners[2]));//
			points.push_back(cv::Point2f(scene_corners[3]));// 3-----2

			getPlanarSurface(points);

			points.clear();
			keypoints1.clear();
			keypoints2.clear();
			matches.clear();

			//Sleep(1000);
			framenum++;
		}
	}
	// return the found fundemental matrix
	return fundemental;
}

int main(int argc, char* argv[])
{

	//Instantiate robust matcher
	RobustMatcher rmatcher;
	bool glog_initialized = false;

	vector<DMatch>  matches;

	bool init(GpuMat img1GPU);    //ly add
	if (!init(img1GPU)) {
		cout << "init error" << endl;
	}
	else {
	}

	for (;;) {

		double t5 = getTickCount();		//ly add

		capture >> frame;
		if (frame.empty())
			break;
		Mat img3;
		frame.copyTo(img3);

		double t6 = getTickCount() - t5;
		t6 = 1000 * t6 / getTickFrequency();
		cout << "framecopy time = " << t6 << " ms " << endl;	//ly add

		clock_t start_time1,end_time1,start_time2,end_time2;

		if (!glog_initialized){
        		glog_initialized = true;
        		//google::InitGoogleLogging("log");
			::google::InitGoogleLogging(argv[0]);
    		}

         	//::google::InitGoogleLogging(argv[0]);

  		string model_file   = "/home/nvidia/project_wly_gpu/deploy.prototxt";
  		string trained_file = "/home/nvidia/project_wly_gpu/wly.caffemodel";
  		string mean_file    = "/home/nvidia/project_wly_gpu/mean.binaryproto";
  		string label_file   = "/home/nvidia/project_wly_gpu/label.txt";
  		start_time1 = clock();
  		Classifier classifier(model_file, trained_file, mean_file, label_file);
  		end_time1 = clock();
  		double seconds1 = (double)(end_time1-start_time1)/CLOCKS_PER_SEC;
  		std::cout<<"init time="<<seconds1<<"s"<<std::endl;

  		//string file = "/home/ubuntu/caffe/project_wly/test_images/6619.jpg";

  		//cv::Mat img = cv::imread(file, -1);
  		std::cout << "---------- Prediction ----------" << std::endl;

  		CHECK(!img3.empty()) << "Unable to decode image " << endl;
 		start_time2 = clock();
 		std::vector<Prediction> predictions = classifier.Classify(img3);
 		end_time2 = clock();
  		double seconds2 = (double)(end_time2-start_time2)/CLOCKS_PER_SEC;
  		std::cout<<"classify time="<<seconds2<<"s"<<std::endl;

  		/* Print the top N predictions. */
  		for (size_t i = 0; i < predictions.size(); ++i) {
    		Prediction p = predictions[i];
    		std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;


			if ( 0.7 <= p.second && p.second <= 1 && p.first == "yes") {

				//cout << "The picture is positive" << endl;
				std::cout << "---------- location ----------" << std::endl;
				frame.copyTo(img2);
				GpuMat img2GPU(img2);
				//imshow("frame", frame);

				rmatcher.match(img1GPU, img2GPU, matches, keypoints1, keypoints2);
				waitKey(2);
			}
			else {
				cout << "The picture is negative" << endl;
			}
		}
	}
	return 0;
}
