# 使用OpenVINO 2021.4 Inference Engine (IE Core) APIs運行影像分類範例

請在命令列環境執行下載及轉換模型動作  
omz_downloader --name alexnet  
omz_converter --name alexnet  
完成後會於分別在 .\public\alexnet\FP16 和 .\public\alexnet\FP32 看到 alexnet.bin 和 alexnet.xml  

banana.jpg 為測試用影像檔  

hello_classification.py 為影像分類範例程式  
執行方式： python hello_classification.py -m ./public/alexnet/FP16/alexnet.xml -i banana.jpg -d GPU  
最後一個參數為推論裝置：CPU, GPU, MYRIAD (即Intel神經加速棒NCS2)

或者直接執行已安排好的批次檔run_cpu.bat, run_gpu.bat, run_vpu.bat來運行範例程式  

