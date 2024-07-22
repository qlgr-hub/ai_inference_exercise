#include "engine.hxx"
#include "model_base.hxx"
#include <cassert>


namespace maie {

static const cv::Size model_input_size{ 224, 224 };

class Resnet50 final : public ModelBase<std::vector<float>> {
    void __softmax(std::vector<float>& x) {
        float max{ 0.f };
        float sum{ 0.f };
        for (size_t k{ 0 }; k < x.size(); ++k) {
            if (max < x[k]) {
                max = x[k];
            }
        }

        for (size_t k{ 0 }; k < x.size(); ++k) {
            x[k] = exp(x[k] - max);
            sum += x[k];
        }

        for (size_t k{ 0 }; k < x.size(); ++k) {
            x[k] /= sum;
        }
    }

    cv::Mat __scaleToFit(const cv::Mat& image, const cv::Size& dst_shape) {
        return image;

        // TODO: need scale ?
        // cv::Size_<float> cur_shape{ float(image.cols), float(image.rows) };
        // cv::Size_<float> new_shape{ float(dst_shape.width), float(dst_shape.height) };
        // auto r{ std::min(new_shape.height / cur_shape.height, new_shape.width / cur_shape.width) };
        // cv::Size_<float> scaled_shape{ std::round(cur_shape.width  * r), std::round(cur_shape.height * r) };
        //
        // // 1. Scale image if need
        // cv::Mat out_image;
        // if (int(cur_shape.width) != int(scaled_shape.width) && int(cur_shape.height) != int(scaled_shape.height)) {
        //     cv::resize(image, out_image, cv::Size(int(scaled_shape.width), int(scaled_shape.height)));
        // }
        // else {
        //     out_image = image.clone();
        // }
        //
        // // 2. Padding
        // cv::Size_<float> pad_size{ new_shape.width - scaled_shape.width, new_shape.height - scaled_shape.height };
        // pad_size.width /= 2.f;
        // pad_size.height /= 2.f;
        // auto bt{ int(std::round(pad_size.height - 0.1f)) };
        // auto bb{ int(std::round(pad_size.height + 0.1f)) };
        // auto bl{ int(std::round(pad_size.width - 0.1f))  };
        // auto br{ int(std::round(pad_size.width + 0.1f))  };
        // cv::copyMakeBorder(out_image, out_image, bt, bb, bl, br, cv::BORDER_CONSTANT, { 144, 144, 144 });
        // return out_image;
    }

private:
    cv::Mat __preprocess(const std::vector<std::string_view>& paths) override {
        std::vector<cv::Mat> images;
        for (auto imagePath : paths) {
            auto scaled_image{ __scaleToFit(cv::imread(imagePath.data()), model_input_size) };
            images.push_back(scaled_image);
        }

        auto blob = cv::dnn::blobFromImages(images, 
            1./ 255., 
            model_input_size, 
            cv::Scalar{123.675, 116.28, 103.53}, 
            true,
            false
        );

        cv::divide(blob, cv::Scalar{0.229, 0.224, 0.225 }, blob);
        return blob;
    }

    bool __postprocess(const DetectResult& result, std::vector<std::vector<float>>& outputs) override {
        assert(result.res_shape.size() == 2);
        outputs.resize(result.res_shape[0]);

        float* output{ result.res };
        auto size_a_batch{ result.res_shape[1] };
        for (int i{ 0 }; i < result.res_shape[0]; ++i) {
            outputs[i].assign(output, output + size_a_batch);
            __softmax(outputs[i]); // to props ?
            output += size_a_batch;
        }

        return true;
    }
};

std::unique_ptr<ImageDetectorResnet50> makeDetectorResnet50() {
    return std::make_unique<Resnet50>();
}

} // namespace maie
