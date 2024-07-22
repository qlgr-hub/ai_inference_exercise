#include "engine.hxx"
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>


namespace maie {

#define CHECK(err) _checkCuda(err, __FILE__, __LINE__)
static inline void _checkCuda(cudaError_t err, const char* file, int32_t line) {
    if (cudaSuccess != err) {
        std::cout << "Cuda error: " << err << " file: " << file << " line: " << line << '\n';
        exit(EXIT_FAILURE);
    }
}


struct Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        //if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
        //    std::cerr << msg << '\n';
        //}
    }
};
static Logger s_logger;


class EngineFileUtils {
    // ref: https://github.com/cyrusbehr/tensorrt-cpp-api/src/engine.h
    static bool __createFromOnnx(std::string_view onnx_model_path, std::string_view engine_file_path) {
        std::unique_ptr<nvinfer1::IBuilder> builder{ nvinfer1::createInferBuilder(s_logger) };
        if (!builder) {
            return false;
        }

        auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(explicitBatch) };
        if (!network) {
            return false;
        }

        std::unique_ptr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, s_logger) };
        if (!parser) {
            return false;
        }

        std::ifstream file(onnx_model_path.data(), std::ios::binary | std::ios::ate);
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            return false;
        }

        auto parsed = parser->parse(buffer.data(), buffer.size());
        if (!parsed) {
            return false;
        }

        std::unique_ptr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
        if (!config) {
            return false;
        }

        const auto input0Batch = network->getInput(0)->getDimensions().d[0];
        bool doesSupportDynamicBatch = (input0Batch == -1) ? true : false;

        // Register a single optimization profile
        nvinfer1::IOptimizationProfile* optProfile = builder->createOptimizationProfile();
        const auto numInputs = network->getNbInputs();
        for (int32_t i = 0; i < numInputs; ++i) {
            // Must specify dimensions for all the inputs the model expects.
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            constexpr auto optBatchSize = 1;
            constexpr auto maxBatchSize = 1;
            int32_t inputC = inputDims.d[1];
            int32_t inputH = inputDims.d[2];
            int32_t inputW = inputDims.d[3];

            // Specify the optimization profile`
            if (doesSupportDynamicBatch) {
                optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
            } else {
                optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                        nvinfer1::Dims4(optBatchSize, inputC, inputH, inputW));
            }
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                    nvinfer1::Dims4(optBatchSize, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                    nvinfer1::Dims4(maxBatchSize, inputC, inputH, inputW));
        }
        config->addOptimizationProfile(optProfile);

        // CUDA stream used for profiling by the builder.
        cudaStream_t profileStream;
        CHECK(cudaStreamCreate(&profileStream));
        config->setProfileStream(profileStream);

        std::unique_ptr<nvinfer1::IHostMemory> modelStream{ builder->buildSerializedNetwork(*network, *config) };
        if (!modelStream) {
            return false;
        }

        std::ofstream f(engine_file_path.data(), std::ios::binary);
        if (f.is_open()) {
            f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        }

        CHECK(cudaStreamDestroy(profileStream));
        return true;
    }

public:
    static bool loadEngineFile(std::string_view onnx_model_path, std::vector<char>& binaryData) {
        namespace fs = std::filesystem;
        
        // 1. Get engine file path from model file path
        fs::path model_file_path{ onnx_model_path };
        fs::path model_file_name{ model_file_path.filename() };
        fs::path engine_file_path{ model_file_path.parent_path().parent_path() };
        engine_file_path /= "engine";
        engine_file_path /= model_file_name;
        engine_file_path.replace_extension(".trt");
        // std::cout << engine_file_path << '\n';

        // 2. If the engine file does not exist, generate it
        if (!fs::exists(engine_file_path)) {
            if (!__createFromOnnx(onnx_model_path, engine_file_path.c_str())) {
                return false;
            }
        }

        // 3. If engine file exists, load it into memory.
        std::ifstream file(engine_file_path.c_str(), std::ios::binary | std::ios::ate);
        if (!file) {
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        binaryData.resize(size);
        if (!file.read(binaryData.data(), size)) {
            return false;
        }

        return true;
    }
};


class EngineTrt final : public IEngine {
    std::unique_ptr<nvinfer1::IRuntime>    __runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> __engine;

    float* __output;

private:
    void __free_output(float*& output) {
        free(output);
        output = nullptr;
    }

public:
    EngineTrt() : __runtime{ nullptr }, __engine{ nullptr }, __output{ nullptr } {
    }

    ~EngineTrt() {
        if (__output) {
            __free_output(__output);
        }
    }

public:
    bool init(std::string_view model_path) override {
        cudaSetDevice(0);

        __runtime.reset(nvinfer1::createInferRuntime(s_logger));
        if (!__runtime) {
            return false;
        }

        std::vector<char> engineData;
        bool ret = EngineFileUtils::loadEngineFile(model_path, engineData);
        if (!ret) {
            return false;
        }

        __engine.reset(__runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!__engine) {
            return false;
        }

        return true;
    }

    bool detect(const cv::Mat& blob, DetectResult& result) override {
        if (!__engine) {
            return false;
        }

        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(__engine->createExecutionContext());
        if (!context) {
            return false;
        }

        // currenly just support one input one output
        assert(__engine->getNbIOTensors() == 2);
        __free_output(__output);

        auto output_size_byes{ 1UL };
        void* devInput{ nullptr };
        void* devOutput{ nullptr };

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        
        for (int i = 0; i < __engine->getNbIOTensors(); ++i) {
            const auto tensorName{ __engine->getIOTensorName(i) };
            const auto tensorType{ __engine->getTensorIOMode(tensorName) };
            const auto tensorShape{ __engine->getTensorShape(tensorName) };
            const auto tensorDataType{ __engine->getTensorDataType(tensorName) };

            if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
                auto input_size_bytes{ blob.total() * blob.elemSize() };
                CHECK(cudaMallocAsync(&devInput, input_size_bytes, stream));
                CHECK(cudaMemcpyAsync(devInput, blob.ptr<float>(), input_size_bytes, cudaMemcpyHostToDevice, stream));

                nvinfer1::Dims4 inputDims{ blob.size.p[0], blob.size.p[1], blob.size.p[2], blob.size.p[3] };
                context->setInputShape(tensorName, inputDims);
                context->setTensorAddress(tensorName, devInput);
            }
            else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
                result.res_shape.push_back(blob.size.p[0]); // set result batch size

                uint32_t osize_a_batch{ 1 };
                for (int j{ 1 }; j < tensorShape.nbDims; ++j) {
                    osize_a_batch *= tensorShape.d[j];
                    result.res_shape.push_back(tensorShape.d[j]);
                }

                output_size_byes = blob.size.p[0] * osize_a_batch * sizeof(float);
                CHECK(cudaMallocAsync(&devOutput, output_size_byes, stream));
                CHECK(cudaMemset(devOutput, 0, output_size_byes));

                context->setTensorAddress(tensorName, devOutput);
            }
        }

        if (!context->allInputDimensionsSpecified()) {
            return false;
        }

        context->enqueueV3(stream);
        
        __output = (float*)malloc(output_size_byes);
        memset(__output, 0, output_size_byes);
        CHECK(cudaMemcpyAsync(__output, devOutput, output_size_byes, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        result.res = __output;

        CHECK(cudaFreeAsync(devInput, stream));
        CHECK(cudaFreeAsync(devOutput, stream));
        CHECK(cudaStreamDestroy(stream));
        return true;
    }
};


std::unique_ptr<IEngine> makeEngineTrt() {
    return std::make_unique<EngineTrt>();
}

} // namespace maie
