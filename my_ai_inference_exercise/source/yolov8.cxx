#include "model_base.hxx"
#include <algorithm>
#include <array>
#include <vector>


namespace maie {

static const cv::Size model_input_size{ 640, 640 };

static constexpr size_t NUM_CLASSES{ 80 };
static constexpr auto PROBABILITY_THRESHOLD{ 0.25f };
static constexpr auto NMS_THRESHOLD{ 0.65f };
static constexpr auto TOP_K{ 100 };


// labels
static std::array<std::string, NUM_CLASSES> s_ClassNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};


class Yolov8 final : public ModelBase<std::vector<OuputYolov8>> {
    struct __imageInfo {
        float org_w;
        float org_h;
        float scale_ratio;
    };

    std::vector<__imageInfo> __image_infos;

private:
    cv::Mat __letterBox(const cv::Mat& image, const cv::Size& dst_shape, bool auto_shape = false, bool scale_fill = false, 
                        bool scale_up = false, int stride = 32, const cv::Scalar& color = { 144, 144, 144 }) {
        // 1. Calculate min scale ratio
        cv::Size_<float> cur_shape{ float(image.cols), float(image.rows) };
        cv::Size_<float> new_shape{ float(dst_shape.width), float(dst_shape.height) };
        auto r{ std::min(new_shape.height / cur_shape.height, new_shape.width / cur_shape.width) };
        if (!scale_up)
            r = std::min(r, 1.0f);
    
        // 2. Calculate scaled shape and padding size
        cv::Size_<float> scaled_shape{ std::round(cur_shape.width  * r), std::round(cur_shape.height * r) };
        cv::Size_<float> pad_size{ new_shape.width - scaled_shape.width, new_shape.height - scaled_shape.height };
        if (auto_shape) {
            pad_size.width = float(int(pad_size.width) % stride);
            pad_size.height = float(int(pad_size.height) % stride);
        }
        else if (scale_fill) {
            pad_size.width = 0.f;
            pad_size.height = 0.f;
            scaled_shape = new_shape;
        }

        // 3. Scale image if need
        cv::Mat out_image;
        if (int(cur_shape.width) != int(scaled_shape.width) && int(cur_shape.height) != int(scaled_shape.height)) {
            cv::resize(image, out_image, cv::Size(int(scaled_shape.width), int(scaled_shape.height)));
        }
        else {
            out_image = image.clone();
        }
    
        // 4. Padding
        pad_size.width /= 2.f;
        pad_size.height /= 2.f;
        auto bt{ int(std::round(pad_size.height - 0.1f)) };
        auto bb{ int(std::round(pad_size.height + 0.1f)) };
        auto bl{ int(std::round(pad_size.width - 0.1f))  };
        auto br{ int(std::round(pad_size.width + 0.1f))  };
        cv::copyMakeBorder(out_image, out_image, bt, bb, bl, br, cv::BORDER_CONSTANT, color);
        
        // used in the post process
        __image_infos.push_back({ cur_shape.width, cur_shape.height, r });

        return out_image;
    }

private:
    cv::Mat __preprocess(const std::vector<std::string_view>& paths) override {
        std::vector<cv::Mat> images;
        for (auto imagePath : paths) {
            auto scaled_image{ __letterBox(cv::imread(imagePath.data()), model_input_size) };
            images.push_back(scaled_image);
        }

        // make blob
        return cv::dnn::blobFromImages(images, 
            1./ 255., 
            model_input_size, 
            cv::Scalar{}, 
            true,
            false
        );
    }

    bool __postprocess(const DetectResult& result, std::vector<std::vector<OuputYolov8>>& outputs) override {
        float* out_engine{ result.res };
        auto size_a_batch{ result.res_shape[1] * result.res_shape[2] };
        for (int n{ 0 }; n < result.res_shape[0]; ++n) {
            // 1. Get a batch result
            cv::Mat output{ cv::Mat(result.res_shape[1], result.res_shape[2], CV_32F, out_engine).t() };
            out_engine += size_a_batch;

            auto ratio{ __image_infos[n].scale_ratio };
            auto image_w{ __image_infos[n].org_w };
            auto image_h{ __image_infos[n].org_h };

            // 2. Get all the YOLO proposals
            std::vector<cv::Rect> bboxes;
            std::vector<float> scores;
            std::vector<int> classIndices;
            for (int i{ 0 }; i < result.res_shape[2]; ++i) {
                constexpr auto BBOX_SIZE{ 4 };
                auto rowPtr{ output.row(i).ptr<float>() };
                auto scoresPtr{ rowPtr + BBOX_SIZE };
                auto maxSPtr{ std::max_element(scoresPtr, scoresPtr + s_ClassNames.size()) };
                auto score{ *maxSPtr };
                if (score > PROBABILITY_THRESHOLD) {
                    auto bboxesPtr{ rowPtr };
                    float x{ *bboxesPtr++ };
                    float y{ *bboxesPtr++ };
                    float w{ *bboxesPtr++ };
                    float h{ *bboxesPtr   };

                    float x0{ std::clamp((x - 0.5f * w) * ratio, 0.f, image_w) };
                    float y0{ std::clamp((y - 0.5f * h) * ratio, 0.f, image_h) };
                    float x1{ std::clamp((x + 0.5f * w) * ratio, 0.f, image_w) };
                    float y1{ std::clamp((y + 0.5f * h) * ratio, 0.f, image_h) };

                    int classIdx{ static_cast<int>(maxSPtr - scoresPtr) };

                    bboxes.push_back({ int(x0), int(y0), int(x1 - x0), int(y1 - y0) });
                    classIndices.push_back(classIdx);
                    scores.push_back(score);
                }
            }

            // 3. Run NMS
            std::vector<int> indices;
            cv::dnn::NMSBoxesBatched(bboxes, scores, classIndices, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

            // 4. Choose the top k detections
            std::vector<OuputYolov8> detections;
            int cnt{ 0 };
            for (auto &chosenIdx : indices) {
                if (cnt >= TOP_K)
                    break;

                OuputYolov8 r{};
                r.confidence = scores[chosenIdx];
                r.cls_name = s_ClassNames[classIndices[chosenIdx]];
                r.range = { bboxes[chosenIdx].x, bboxes[chosenIdx].y, bboxes[chosenIdx].width, bboxes[chosenIdx].height };
                detections.emplace_back(r);

                cnt += 1;
            }

            outputs.emplace_back(detections);
        }
        return true;
    }
};

std::unique_ptr<ImageDetectorYolov8> makeDetectorYolov8() {
    return std::make_unique<Yolov8>();
}

} // namespace maie