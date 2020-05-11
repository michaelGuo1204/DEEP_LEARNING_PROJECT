#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<vector>
#include<fstream>
using namespace cv;
using namespace std;
using namespace cv::ml;
class Gene
{

public:
    Gene(std::string One_line_data);
    ~Gene();
     cv::Mat gene_data;
    int label;
};

Gene::Gene(std::string One_line_data)
{
    const char* data=One_line_data.c_str();
for(int i=0;i<228;i++)
{
    switch (data[i])
    {
    case 'T':
         gene_data.push_back(0);
        break;
    case 'A':
        gene_data.push_back(1);
        break;
    case 'C':
        gene_data.push_back(2);
        break;
    case 'G':
        gene_data.push_back(3);
        break;
    default:
        break;
    }
}
    sscanf(data,"%*s %d",&label);
}

Gene::~Gene()
{
}


int main()
{

    //cout<<cv::getVersionString()<<endl;  OpenCV version 3.4.7 confirmed
    ifstream fs;
    char buffer[250];
    fs.open("input-output.txt");
    vector<Gene> input_data;
    if(fs.is_open())
    {
        while (fs.good()&&!fs.eof())
        {
           memset(buffer,0,250);
           fs.getline(buffer,250);
           string buffer_string =buffer;
           Gene cache(buffer_string);
           input_data.push_back(cache);
        }
        fs.close();
    }
    cv::Mat Sample;
    cv::Mat label(cv::Size(0,0),CV_32SC1);
    int count=0;
    cv::Mat Test_sample;
    cv::Mat Test_label(cv::Size(0,0),CV_32SC1);
    for(auto &element :input_data)
    {
        if(element.gene_data.rows==228){
        cv::resize(element.gene_data,element.gene_data,Size(1,228),(0,0),(0,0),INTER_AREA);
        element.gene_data.convertTo(element.gene_data,CV_32FC1);
        if(count<1307)
        {Sample.push_back(element.gene_data.reshape(0,1));
        label.push_back<int>(element.label);
        }
        else{
            Test_sample.push_back(element.gene_data.reshape(0,1));
        Test_label.push_back<int>(element.label);
        }
        count++;
        }
    }
    cout<<"loading complete"<<endl;
    Ptr<KNearest>knn =KNearest::create();
    knn->setDefaultK(5);
    knn->setIsClassifier(true);
    cv::Ptr<cv::ml::TrainData> &&train_set=cv::ml::TrainData::create(Sample,cv::ml::ROW_SAMPLE,label);
    cout<<"training"<<endl;
    knn->train(train_set);

    int current=0;
    count=0;
    for (int i = 0; i < Sample.rows; i++)
    {
        Mat test_cache;
        Sample.row(i).copyTo(test_cache);
        test_cache.convertTo(test_cache,CV_32FC1);
        float r=knn->predict(test_cache.reshape(0,1));
        if(((int)r) ==(label.at<int>(i,0)))current++;
        count++;
    }
    float currency=(int)(current*100/count);
    cout<<"Accurancy is "<<currency<<"%"<<endl;
}
