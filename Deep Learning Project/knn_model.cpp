#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<vector>
#include<fstream>
using namespace cv;
using namespace std;
using namespace cv::ml;
class Case
{

public:
    Case(string label,std::string One_line_data,int status);
    ~Case();
    cv::Mat _Gene_data;
    string _label;
    int _status;
private:
    int Get_number(const char* allo);
};
int Case::Get_number(const char* allo)
{
    int Gene_cell[4]={0,0,0,0};
for(int i=0;i<4;i+=2){    
    switch (allo[i])
    {
    case 'A':
        Gene_cell[0]+=1;
        break;
    case 'T':
        Gene_cell[1]+=1;
        break;
    case 'C':
        Gene_cell[2]+=1;
        break;
    case 'G':
        Gene_cell[3]+=1;
    default:
        break;
    }
}
return Gene_cell[0]*1000+Gene_cell[1]*100+Gene_cell[2]*10+Gene_cell[3];
}
Case::Case(string label,std::string One_line_data,int status)
{
    _label=label;
    const char* data=One_line_data.c_str();
for(int i=0;i<One_line_data.length();i+=4)
{
   // cout<<Get_number(&data[i])<<endl;
_Gene_data.push_back(Get_number(&data[i]));
}
    _status=status;
}

Case::~Case()
{
}


int main()
{

    //cout<<cv::getVersionString()<<endl;  OpenCV version 3.4.7 confirmed
    ifstream status_stream;
    ifstream gene_stream;
    char buffer_gene[2900],buffer_status[100];
    status_stream.open("/media/bili/L/Bioinformatics/TypeII Diabetes/type 2 diabetes Raw Data/all-ids-phe");
    gene_stream.open("/media/bili/L/Bioinformatics/TypeII Diabetes/type 2 diabetes Raw Data/all-seq");
    vector<Case> input_data;
    if(status_stream.is_open()&&gene_stream.is_open())
    {
        while (status_stream.good()&&!status_stream.eof()&&gene_stream.good()&&!gene_stream.eof())
        {
            string label;int status_num;
            int label_length;
           memset(buffer_gene,0,2900);
           memset(buffer_status,0,100);
           gene_stream.getline(buffer_gene,2900);
        status_stream.getline(buffer_status,100);
        string status_str=buffer_status;
        string gene_str=buffer_gene;
        label_length=status_str.find("\t");
        string status_s_str=status_str.substr(0,label_length);
        string gene_s_str=gene_str.substr(0,label_length);
        string gene_oneline=gene_str.substr(label_length+2);
        label=status_s_str;
        status_num=*(status_str.end()-1)=='\r'?(int)*(status_str.end()-2)-'0':(int)*(status_str.end()-1)-'0';
        Case _cache_case(label,gene_oneline,status_num);
        input_data.push_back(_cache_case);
        }
        status_stream.close();
        gene_stream.close();
    }

    cv::Mat Sample;
    cv::Mat label(cv::Size(0,0),CV_32SC1);
    int count=0;
    cv::Mat Test_sample;
    cv::Mat Test_label(cv::Size(0,0),CV_32SC1);
    for(auto &element :input_data)
    {
        if(element._Gene_data.rows==722){
        cv::resize(element._Gene_data,element._Gene_data,Size(1,722),(0,0),(0,0),INTER_AREA);
        element._Gene_data.convertTo(element._Gene_data,CV_32FC1);
        if(count<7000)
        {Sample.push_back(element._Gene_data.reshape(0,1));
        label.push_back<int>(element._status);
        }
        else{
            Test_sample.push_back(element._Gene_data.reshape(0,1));
        Test_label.push_back<int>(element._status);
        }
        count++;
        }
    }
    Ptr<KNearest>knn =KNearest::create();
    knn->setDefaultK(5);
    knn->setIsClassifier(true);
    cv::Ptr<cv::ml::TrainData> &&train_set=cv::ml::TrainData::create(Sample,cv::ml::ROW_SAMPLE,label);
    cout<<"training"<<endl;
    knn->train(train_set);
    knn->save("./knn.xml");
    knn->load("./knn.xml");
    int current=0;
    count=0;
    for (int i = 0; i < Test_sample.rows; i++)
    {
        Mat test_cache;
        Test_sample.row(i).copyTo(test_cache);
        test_cache.convertTo(test_cache,CV_32FC1);
        float r=knn->predict(test_cache.reshape(0,1));
        //cout<<Test_label.at<int>(i,0)<<endl;
        if(((int)r) ==(Test_label.at<int>(i,0)))current++;
        count++;
    }
    float currency=(int)(current*100/count);
    cout<<"Accurancy is "<<currency<<"%"<<endl;
}
