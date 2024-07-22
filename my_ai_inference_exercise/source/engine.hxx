#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>


namespace maie {

struct DetectResult {
    float* res; // result tensor's buffer address
    std::vector<int64_t> res_shape; // result tensor's shape
};


struct IEngine {
    virtual bool init(std::string_view model_path) = 0;
    virtual bool detect(const cv::Mat& blob, DetectResult& result) = 0;
};


std::unique_ptr<IEngine> makeEngineOrt();
std::unique_ptr<IEngine> makeEngineTrt();

} // namespace maie
