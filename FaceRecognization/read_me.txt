此文件夹为人脸识别的程序包
此程序再Ubuntu16.04环境下进行的测试。
先运行faceRecognization.py将训练好的参数存到data.pkl中。
再运行video.py，一段时间后关闭此窗口（至少要保证存在绿色方框），文件夹中会产生一个video.avi
接着运行usingPkl.py会出现结果，分别对应着deal里的每一张图片。