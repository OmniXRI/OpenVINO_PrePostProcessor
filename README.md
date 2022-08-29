# OpenVINO_PrePostProcessor
比較OpenVINO使用傳統Inference Engine和新版PrePostProcessor預處理模型推論效能範例

OpenVINO在2022.1版後使用用Runtime APIs取代原有的Inference Engine(IE Core) APIs，
同時加入PrePostProcessor，讓iGPU, VPU也能協助影像預處理及推論後處理，
其概念如圖所示。

<img src="https://github.com/OmniXRI/OpenVINO_PrePostProcessor/blob/main/images/20220822_Fig_01.jpg" width="640">

這裡提供幾個測試案例：  
\hello_classification_ie 使用IE Core進行推論  
\hello_classification_ov 使用Runtime Core進行推論  
\contrast_opencv_ppp 亮度/對比增強（減弱）使用OpenCV, OpenVINO PrePostProcessor API進行比較  

完整說明文件，請參考「有了OpenVINO 2022 PrePostProcessor APIs影像推論就更有效率了」  
https://omnixri.blogspot.com/2022/08/openvino-2022-prepostprocessor-apis.html
