# Augmentation Feature Roadbook

Hi there! For those who are interested in testing 
augmentation techniques in `Time-Series-Library`.

For now, we have embedded several augmentation methods
in this repo. We are still collecting publicly available 
augmentation algorithms, and we appreciate your valuable
advice!

```
The Implemented Augmentation Methods
1. jitter 
2. scaling 
3. permutation 
4. magwarp 
5. timewarp 
6. windowslice 
7. windowwarp 
8. rotation 
9. spawner 
10. dtwwarp 
11. shapedtwwarp 
12. wdba (Specially Designed for Classification tasks)
13. discdtw
```

## Usage

In this folder, we present two sample of shell scripts 
doing augmentation in `Forecasting` and `Classification`
tasks.

Take `Forecasting` task for example, we test multiple
augmentation algorithms on `EthanolConcentration` dataset
(a subset of the popular classification benchmark `UEA`) 
using `PatchTST` model.

```shell
 

model_name=PatchTST

for aug in jitter scaling permutation magwarp timewarp windowslice windowwarp rotation spawner dtwwarp shapedtwwarp wdba discdtw discsdtw
do
echo using augmentation: ${aug}

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --augmentation_ratio 1 \
  --${aug}
 done
```

Here, parameter `augmentation_ratio` represents how many
times do we want to perform our augmentation method.
Parameter `${aug}` represents a string of augmentation
type label. 

The example here only perform augmentation once, so we
can set `augmentation_ratio` to `1`, followed by one
augmentation type label. Trivially, you can set 
`augmentation_ratio` to an integer `num` followed by 
`num` augmentation type labels.

The augmentation code obeys the same prototype of 
`Time-Series-Library`. If you want to adjust other 
training parameters, feel free to add arguments to the
shell scripts and play around. The full list of parameters
can be seen in `run.py`.

## Contact Us!

This piece of code is written and maintained by 
[Yunzhong Qiu](https://github.com/DigitalLifeYZQiu). 
We thank [Haixu Wu](https://github.com/wuhaixu2016) and
[Jiaxiang Dong](https://github.com/dongjiaxiang) for 
insightful discussion and solid support.

If you have difficulties or find bugs in our code, please
contact us:
- Email: qiuyz24@mails.tsinghua.edu.cn

# 增强功能路线图

你好！对于那些对测试 `Time-Series-Library` 中的增强技术感兴趣的人。

目前，我们在这个仓库中嵌入了几种增强方法。我们仍在收集公开可用的增强算法，感谢您的宝贵建议！

```
已实现的增强方法
1. 抖动
2. 缩放
3. 排列
4. 幅度扭曲
5. 时间扭曲
6. 窗口切片
7. 窗口扭曲
8. 旋转
9. 生成器
10. 动态时间规整扭曲
11. 形状动态时间规整扭曲
12. 加权动态时间规整（专为分类任务设计）
13. 离散动态时间规整
```

## 用法

在此文件夹中，我们提供了两个示例 shell 脚本，用于在 `预测` 和 `分类` 任务中进行增强。

以 `预测` 任务为例，我们在 `EthanolConcentration` 数据集（流行的分类基准 `UEA` 的子集）上使用 `PatchTST` 模型测试多种增强算法。

```shell
 

model_name=PatchTST

for aug in jitter scaling permutation magwarp timewarp windowslice windowwarp rotation spawner dtwwarp shapedtwwarp wdba discdtw discsdtw
do
echo using augmentation: ${aug}

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --augmentation_ratio 1 \
  --${aug}
 done
```

这里，参数 `augmentation_ratio` 表示我们希望执行增强方法的次数。参数 `${aug}` 表示增强类型标签的字符串。

此示例仅执行一次增强，因此我们可以将 `augmentation_ratio` 设置为 `1`，后跟一个增强类型标签。简单地说，您可以将 `augmentation_ratio` 设置为整数 `num`，后跟 `num` 个增强类型标签。

增强代码遵循 `Time-Series-Library` 的相同原型。如果您想调整其他训练参数，请随意向 shell 脚本添加参数并进行尝试。参数的完整列表可以在 `run.py` 中看到。

## 联系我们！

这段代码由 [Yunzhong Qiu](https://github.com/DigitalLifeYZQiu) 编写和维护。我们感谢 [Haixu Wu](https://github.com/wuhaixu2016) 和 [Jiaxiang Dong](https://github.com/dongjiaxiang) 的深刻讨论和坚实支持。

如果您在使用我们的代码时遇到困难或发现错误，请联系我们：
- Email: qiuyz24@mails.tsinghua.edu.cn