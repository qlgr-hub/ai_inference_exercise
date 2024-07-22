#include "engine.hxx"
#include <onnxruntime_cxx_api.h>


namespace maie {

class EngineOrt final : public IEngine {
    Ort::SessionOptions __session_options;
    std::unique_ptr<Ort::Session> __session;

    std::vector<Ort::Value> output_tensors;

public:
    EngineOrt() : __session{ nullptr } {
        // TODO: It is necessary to provide an interface for setting options.
        __session_options.SetIntraOpNumThreads(16);
        __session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

public:
    bool init(std::string_view model_path) override {
        // If defined as a local variable, Segmentation fault will occur during inference
        // If defined as a member variable, initialization is very inconvenient
        static Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "EngineOrt" };
        __session = std::make_unique<Ort::Session>(env, model_path.data(), __session_options);
        return (__session != nullptr);
    }

    bool detect(const cv::Mat& blob, DetectResult& result) override {
        if (!__session) {
            return false;
        }

        auto memory_info{ 
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault) 
        };

        // input dims, get the corresponding size from the blob for dynamic shape
        auto input_dims { __session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() };
        if (input_dims[0] == -1)
            input_dims[0] = blob.size.p[0];
        if (input_dims[1] == -1)
            input_dims[1] = blob.size.p[1];
        if (input_dims[2] == -1)
            input_dims[2] = blob.size.p[2];
        if (input_dims[3] == -1)
            input_dims[3] = blob.size.p[3];

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, 
                const_cast<float*>(blob.ptr<float>()), 
                blob.total(), 
                input_dims.data(), 
                input_dims.size()
            ));

        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name{ __session->GetInputNameAllocated(0, allocator).get() };
	    std::string output_name{ __session->GetOutputNameAllocated(0, allocator).get() };
        std::vector<const char*> input_names{ input_name.c_str() };
        std::vector<const char*> output_names{ output_name.c_str() };

        output_tensors = __session->Run(Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensors.data(),
            __session->GetInputCount(),
            output_names.data(),
            __session->GetOutputCount()
        );

        result = {
            output_tensors[0].GetTensorMutableData<float>(),
            output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()
        };

        return true;
    }
};


std::unique_ptr<IEngine> makeEngineOrt() {
    return std::make_unique<EngineOrt>();
}

} // namespace maie

