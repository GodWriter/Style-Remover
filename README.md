# Style-Remover
A neural network to remove the images' specific style



### 2019/7/19

> 需求分析

* 建立dataloader，考虑数据的加载方式
  * 原始图片和生成图片加载到同一个tf-record中
  * 通过文件名寻找图片，并 创建tf-record
* 模型搭建
  * 主要模式仿照快速风格迁移的模型
    * 输入：带风格的图
    * 输出：生成原图
  * 生成原图和真实原图，输入到预训练好的Vgg模型中，计算内容损失
  * 更新去除风格模型
* 在实验中对比两种方法的生成图片，突出优势



### 2019/7/22

> 下一步

* dataloader已创建完，下一步就是搭建模型
* 模型搭建
  * 主要还是一样的网络结构