## AI inference exercise
* Engine
  * onnxruntime
  * TensorRT
* model
  * resnet50
  * yolov8n
### Build
* build TensorRT
  * Refer to [TensorRT](https://github.com/NVIDIA/TensorRT.git)'s README
* build OpenCV
  * Refer to [here](https://github.com/cyrusbehr/tensorrt-cpp-api/blob/main/scripts/build_opencv.sh)
* build onnxruntime
  * Refer to [here](https://onnxruntime.ai/docs/build/inferencing.html)
* build my_ai_inference_exercise
  * Use build.sh (Can specify Debug or Release via parameters)
### Examples
* detectresnet50 & detectyolov8
  * params: -m \<model path\> -i \<image path\> -e \[TRT|ORT\]
* In the current repository, only my_ai_inference_exercise/images/dog.jpg is available for resnet50 inference.
### Structure
```
workspace/
├── docker
├── my_ai_inference_exercise/
│   ├── engine/     # TensorRT engine file path
│   ├── examples/   # some examples
│   ├── images/     # some images for detect
│   ├── include/
│   ├── labels/     # resnet50 label file path
│   ├── models/     # onnx model file path
│   ├── source/ 
├── third_party/
│   ├── nlohmann/   # for load json file
│   ├── build.sh    # build scripts
│   ├── CMakeLists.txt
└── README.md
```
### Diagram
![Class Diagram](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/qlgr-hub/ai_inference_exercise/main/my_ai_inference_exercise/uml/basic_structure.puml)
### References
* https://github.com/NVIDIA/TensorRT.git
* https://github.com/microsoft/onnxruntime.git
* https://github.com/ultralytics/ultralytics.git
* https://github.com/nlohmann/json.git
* https://github.com/cyrusbehr/tensorrt-cpp-api.git
* https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP.git
* https://github.com/Zingam/UML-in-Markdown.git
* https://pdf.plantuml.net/1.2019.9/PlantUML_Language_Reference_Guide_zh.pdf
* https://blog.csdn.net/weixin_45824067/article/details/130514799