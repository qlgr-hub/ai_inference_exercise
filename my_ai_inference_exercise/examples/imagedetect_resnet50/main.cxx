#include <cassert>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>
#include "../../include/model.hxx"
#include "nlohmann/json.hpp"


struct Paths {
    std::string model_path;
    std::string image_path;
    std::string label_path;
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

    namespace fs = std::filesystem;
    fs::path label_path{ fs::path{model_path}.parent_path().parent_path() };
    label_path /= "labels";
    label_path /= "imagenet-simple-labels.json";

    return { model_path, image_path, label_path.c_str(), engine_path };
}


template<typename T> 
static std::vector<int> indexsort(const std::vector<T>& arr, bool reverse = false) {
    std::vector<int> arrIdx(arr.size(), 0);
    for (int i = 0; i < arr.size(); ++i) {
        arrIdx[i] = i;
    }

    if (reverse) {
        std::sort(arrIdx.begin(), arrIdx.end(), [&arr](size_t pos1, size_t pos2) { 
            return (arr[pos1] > arr[pos2]); 
        });
    }
    else {
        std::sort(arrIdx.begin(), arrIdx.end(), [&arr](size_t pos1, size_t pos2) { 
            return (arr[pos1] < arr[pos2]); 
        });
    }

    return arrIdx;
}


static void getLabels(std::string_view label_path, std::vector<std::string>& labels) {
    nlohmann::json jsonArray;
    std::ifstream jFile{ label_path.data() };
    jFile >> jsonArray;
    labels = jsonArray.get<std::vector<std::string>>();
}


int main(int argc, char* argv[]) {
    auto [model_path, image_path, label_path, engine_path] = parseParams(argc, argv);

    if (model_path.empty() || image_path.empty()) {
        std::cout << "Usage: " << argv[0] << " -m <model_path> -i <image_path> -e [ORT|TRT]\n";
        return 1;
    }

    auto detector{ maie::makeDetectorResnet50() };
    if (!detector) {
        std::cout << "maie::makeDetectorResnet50() fail\n";
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

    std::vector<std::vector<float>> outputs;
    if (!detector->detect({ image_path }, outputs)) {
        std::cout << "detector detect fail\n";
        return 1;
    }

    assert(outputs.size() == 1);
    std::vector<int> argRes = indexsort(outputs[0], true);

    std::vector<std::string> labels{};
    getLabels(label_path, labels);

    std::cout << "Top 5 Predictions:\n";
    for (size_t i{ 0 }; i < 5; ++i) {
        std::cout << labels[argRes[i]] << ", [" << outputs[0][argRes[i]] << "]\n";
    }

    return 0;
}