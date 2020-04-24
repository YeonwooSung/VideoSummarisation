# VideoSummarisation

Video summarisation system that uses the object detection (YOLOv3) and action detection (Epic Kitchens).

This repository contains the codes for my SH project in University of St Andrews.

## Install requirements

First, install all libraries in the requirements.txt.

```bash
pip3 install -r requirements.txt
```

Then, you need to install the pretrainedmodel.

```bash
pip3 install git+https://github.com/wpwei/pretrained-models.pytorch.git@vision_bug_fix
```

This is because that the EpicKitchens action detection model is dependent upon a [fork of Remi Cadene's pretrained models](https://github.com/wpwei/pretrained-models.pytorch/treevision_bug_fix) that brings `DataParallel` support to PyTorch 1+.
