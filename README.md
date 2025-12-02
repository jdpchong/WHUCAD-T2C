data文件夹尚未上传
数据结构类似：
    project/
      data/
    vec/
      0000/00000000.h5
      0000/00000001.h5
      ...
    text/
      0000/00000000.txt
      0000/00000001.txt
      ...
    json/
      0000/00000000.json
      0000/00000001.json

其中vec是h5文件，text是模型描述，json是vec通过deta-pre代码得到的json文件（命令流）

本项目的研究目标是实现基于whucad的text2cad的工作

在项目根目录运行：
python train_text2whucad.py --data_root data --batch_size 8 --epochs 10
训练完成后，你会在 checkpoints/ 里得到一个 text2whucad_best.pt，里面包含模型权重和 vocab
