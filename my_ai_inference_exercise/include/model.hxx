#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>


namespace maie {

enum class EngineType {
    ET_Ort = 0,
    ET_Trt
};


struct OuputYolov8 {
    std::string cls_name;
    float confidence;

    struct Range {
        int x;
        int y;
        int w;
        int h;
    } range;
};


template <typename OT>
struct ImageDetector {
    virtual bool init(std::string_view model_path, EngineType engine_type) = 0;
    virtual bool detect(const std::vector<std::string_view>& paths, std::vector<OT>& outputs) = 0;
};


using ImageDetectorResnet50 = ImageDetector<std::vector<float>>;
using ImageDetectorYolov8 = ImageDetector<std::vector<OuputYolov8>>;

std::unique_ptr<ImageDetectorResnet50> makeDetectorResnet50();
std::unique_ptr<ImageDetectorYolov8> makeDetectorYolov8();

} // namespace maie
