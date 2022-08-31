# 使用OpenVINO 2022.1 Runtime APIs運行影像分類範例

請在命令列環境執行下載及轉換模型動作  
omz_downloader --name alexnet  
omz_converter --name alexnet  
完成後會於分別在 .\public\alexnet\FP16 和 .\public\alexnet\FP32 看到 alexnet.bin 和 alexnet.xml

banana.jpg 為測試用影像檔  

hello_classification.py  為使用PrePostProcessor (PPP) APIs產生預處理模型及執行範例程式，執行後會產生PPP預處理模型IR檔（ppp_model_saved.binppp_model_saved.xml）。  
執行方式： python hello_classification.py ./public/alexnet/FP16/alexnet.xml banana.jpg CPU  
最後一個參數為推論裝置：CPU, GPU, MYRIAD (即Intel神經加速棒NCS2)  

hello_classification_ppp.py 為直接讀取PPP預處理模型進行推論範例程式  
執行方式： python hello_classification_ppp.py ppp_model_saved.xml banana.jpg CPU  

若有需要單獨測試模型在不同裝置上推論效能，可使用benchmark_app來測試。  
-d 為推論裝置(CPU, GPU, MYRIAD)， -api 指定同步或非同步運行，這裡指定為sync同步運行， -t 為測試秒數，若不指定預設為60秒。   
原始模型： benchmark_app -m .\public\alexnet\FP16\alexnet.xml -d CPU -api sync -t 10  
預處理模型： benchmark_app -m ppp_model_saved_cpu.xml -d CPU -api sync -t 10  

