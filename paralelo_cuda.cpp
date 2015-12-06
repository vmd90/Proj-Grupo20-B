#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>

void exec_smooth(const unsigned char* img, unsigned char* out, int width, int height, int step);

int main(int argc, char *argv[])
{
    if(argc < 3) {
        std::cout << "Forma de executar:\n./paralelo_cuda img1 img2" << std::endl;
        std::cout << "\nonde img1 - imagem de entrada"
                     "\n     img2 - imagem de saida" << std::endl;
        return 1;
    }
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if(!img.data) {
        std::cerr << "Erro ao abrir imagem." << std::endl;
        return 1;
    }

    std::vector<unsigned char> img_out(img.cols*img.rows);

    std::vector<cv::Mat> ch(img.channels());
    cv::split(img, ch);

    std::vector<cv::Mat> out(img.channels());
    clock_t t1, t2;
    t1 = clock();
    for(int i = 0; i < ch.size(); ++i)
    {
        std::cout << "Image rows: " << img.rows << ", cols: " << img.cols << ", step: "<< img.step;
        exec_smooth(img.data, &img_out[0], img.rows, img.cols, img.step);
        out[i] = cv::Mat(img.rows, img.cols, CV_8UC1, &img_out[0]);
    }
    t2 = clock();

    std::cout << "\nTempo: " << ((float)t2 - t1)/CLOCKS_PER_SECOND << " s" << std::endl;
    
    cv::Mat m;
    cv::merge(out, m);
    cv::imwrite(argv[2], m);
    
    return 0;
}
