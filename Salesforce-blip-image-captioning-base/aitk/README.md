# blip-image-captioning-base Model Optimization

This repository demonstrates the optimization of the [blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) model using **post-training quantization (PTQ)** techniques. The optimization process is divided into these workflows:

- OpenVINO for Intel® NPU
   + This process uses OpenVINO specific passes like `OpenVINOOptimumConversion`, `OpenVINOIoUpdate` and `OpenVINOEncapsulation`

## Intel® Workflows

These workflows performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*