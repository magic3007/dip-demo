# dip-demo
⚖️ Demo Implementation for the lesson Digital Image Processing(2022 Spring, advised by Yuxin Peng) in Peking University.


## Usage

### Demo System
在`当前目录`下运行如下命令
```bash
python app.py
```
按照提示的网址(如http://127.0.0.1:5000/), 在浏览器中打开，就得到了展示界面, 共有3个小作业的进入按钮, 在每个小作业页面中, 根据网页提示上传图片, 即可得到处理后的图片效果.

### Scripts
另外, 在`scripts`目录下分别提供了三个按照作业要求运行多个图像文件的脚本,
```bash
cd scripts
python HistEq_test.py
python Morph_test.py
python Laplace_test.py
```
注意为了方便助教测试, 我们在上面的三个运行脚本里写了数据文件的相对路径, 可以直接在提交文件的目录环境中进行测试, 如果需要更改输入文件路径, 在文件内更改对应变量即可. 
另外, 输出路径也是在文件内指定的, 目前设置的分别是`./HistEq_test`, `./Morph_test/` 和 `./Laplace_test.py`. 运行完后可以直接在上述目录下看到运行结果.