#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <unistd.h>
#include "../../include/model.hxx"


struct Paths {
    std::string model_path;
    std::string image_path;
    std::string engine_path;
};


static Paths parseParams(int argc, char* argv[]) {
    std::string model_path;
    std::string image_path;
    std::string engine_path;

    int c{ -1 };
    while ((c = getopt(argc, argv, "m:i:e:")) != -1) {
        switch (c) {
        case 'm': model_path = optarg;  break;
        case 'i': image_path = optarg;  break;
        case 'e': engine_path = optarg; break;
        default: break;
        }
    }
    return { model_path, image_path, engine_path };
}


int main(int argc, char* argv[]) {
    auto [model_path, image_path, engine_path] = parseParams(argc, argv);

    if (model_path.empty() || image_path.empty()) {
        std::cout << "Usage: " << argv[0] << " -m <model_path> -i <image_path> -e [ORT|TRT]\n";
        return 1;
    }

    auto detector{ maie::makeDetectorYolov8() };
    if (!detector) {
        std::cout << "maie::makeDetectorYolov8() fail\n";
        return 1;
    }

    maie::EngineType eType{ maie::EngineType::ET_Ort };
    if (engine_path == "TRT") {
        eType = maie::EngineType::ET_Trt;
    }

    if (!detector->init(model_path, eType)) {
        std::cout << "detector init fail\n";
        return 1;
    }

    std::vector<std::vector<maie::OuputYolov8>> outputs;
    if (!detector->detect({ image_path }, outputs)) {
        std::cout << "detector detect fail\n";
        return 1;
    }

    // print all result
    for (const auto topk : outputs[0]) {
        std::cout << topk.cls_name << ", [" << topk.confidence << "]\n"; 
    }

    return 0;
}