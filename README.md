# MLFusion

This is the official code for the paper "MLFuse: Multi-scenario Feature Joint Learning for Multi-Modality Image Fusion".

## Environment Preparing
```shell
python 3.8
pytorch 1.7.0
torchvision 0.8.1
```

### For training
* We provide some data for training in `./train_dataset/`
* For a quick start, please run 
```shell
python train.py --root './train_dataset' --batch_size 16 --save_path './train_result' --summary_name 'MultiTask_qiuck_start_'
```

### For testing
* We provide some example images for testing in `./test_dataset/`
* For a quick start, please run 
```shell
python test_gray.py --model_path './model/model_fuse.pth' --test_path './test_dataset/ct_mri_set' --result_path './test_result/test_ct'
```
* Managing RGB Input

    We refer to the [code of hanna-xu](https://github.com/hanna-xu/utils/tree/master/fusedY2RGB) to convert the fused image into a color image.
    The corresponding code file is 'YCbCr2RGB_Main_Double.m'
### Reference
If you find our work useful in your research please consider citing our paper:
```
@article{lei2023galfusion,
  title={GALFusion: Multi-exposure Image Fusion via a Global-local Aggregation Learning Network},
  author={Lei, Jia and Li, Jiawei and Liu, Jinyuan and Zhou, Shihua and Zhang, Qiang and Kasabov, Nikola K},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```