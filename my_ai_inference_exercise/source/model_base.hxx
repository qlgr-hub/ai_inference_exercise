#pragma once

#include "../include/model.hxx"
#include "engine.hxx"


namespace maie {

template <typename OT>
class ModelBase : public ImageDetector<OT> {
    std::unique_ptr<IEngine> __engine;

private:
    virtual cv::Mat __preprocess(const std::vector<std::string_view>& paths) = 0;
    virtual bool __postprocess(const DetectResult& engineResult, std::vector<OT>& outputs) = 0;

public:
    ModelBase() : __engine{ nullptr } {
    }

public:
    bool init(std::string_view model_path, EngineType engine_type) override {
        switch (engine_type) {
        case EngineType::ET_Ort: __engine = makeEngineOrt(); break;
        case EngineType::ET_Trt: __engine = makeEngineTrt(); break;
        }

        return (__engine && __engine->init(model_path));
    }

    bool detect(const std::vector<std::string_view>& paths, std::vector<OT>& outputs) override {
        // 1. preprocess (Needs optimization, currently using CPU)
        cv::Mat blob{ __preprocess(paths) };

        // 2. inference
        DetectResult result;
        if (!__engine || !__engine->detect(blob, result)) {
            return false;
        }

        // 3. postprocess
        return __postprocess(result, outputs);
    }
};

} // namespace maie