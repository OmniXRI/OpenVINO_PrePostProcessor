# 比較使用OpenCV & OpenVINO PrePostProcessor (ppp) APIs運行亮度/對比增強像素處理速度差異

scenery.jpg 測試用影像  

ov_opencv_linear.py 使用OpenCV & Numpy函式運行線性亮度/對比增強像素處理範例程式  
執行方式： python ov_opencv_linear.py scenery.jpg  
產生結果： contrast_opencv_linear.jpg  

ov_opencv_linear.py 使用OpenCV & Numpy函式運行非線性亮度/對比增強像素處理範例程式  
執行方式： python ov_opencv_nonlinear.py scenery.jpg  
產生結果： contrast_opencv_nonlinear.jpg  

ov_ops.py  根據參考文獻建立之OpenVINO 2022.1 PPP範例程式（含產生預處理PPP IR模型simple_model_saved.xmlsimple_model_saved.bin）  
執行方式：

ov_custom_ppp.py 








實驗結果示意圖：  
![](https://github.com/OmniXRI/OpenVINO_PrePostProcessor/blob/main/images/20220824_Fig_04.jpg?raw=true)

參考文獻：  
Intel, Image Preprocessing with intel Distribution of OpenVINO Toolkit Pre-/Post-Processor APIs White paper (undefined)  
https://cdrdv2.intel.com/v1/dl/getContent/730425?explicitVersion=true
