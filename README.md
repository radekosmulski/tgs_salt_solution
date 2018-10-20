**For a solution overview that placed 9th with very nice pytorch models and clean code, check out this [write up](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053) and the accompanying github repository.**

This repository contains code that I used for my solution to the [Kaggle TGS Salt competition](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69095).

I had good results with taking means of predections of three models trained over 5 folds. This got me to 0.856 on public LB. I than started to train on 10 folds and just the averaged predictions of my best performing model, unet based on incv4, got me to 0.858. If I only take predictions of the 6 best performing folds, I get to 0.86.

This code is nothing more than a draft. I started working on this competition late and I wrote it on whatever spare time I was able to find. The code has issues - some of which I am aware of. For instances, the implementation of the spatial and channel-wise SE is incorrect. Unet based on se resnext 50 has an obvious design issue. I am not applying brightness / contrast / gamma augmentation during training nor during test time. And last but not least, I am applying scSE only in the paths leading to the decoder. According to the paper (and what makes more sense if you think about it) they should be applied in the downstream and upstream paths (the skip connections would automatically be included).

Writing code (same with learning) is an iterative process and it is okay that at this point the code looks how it does. Have I had more time to work on this, the code would look differently. This is true for any code one writes and I don't think ML is in any way unique in this regard.

Why am I sharing this code if it has all the shortcomings? Because the general approach is good. And some of the code can be a good starting point for working on similar problems.

One thing I have not done enough was looking at the data and error analysis. I have done some of that but it is not shown here, should have spent way more time on that.

I am also including code for a Full-Resolution Residual Network - I haven't had much success with training it for this competition. Maybe the code has issues. Should really be tested on the dataset from the paper.

I am also including code for solving the jigsaw problem (identifying neighbors). I think this can be quite useful in many senarios and the idea of taking a ratio of best two candidates is really neat. I am also using [annoy](https://github.com/spotify/annoy) which is a gem of a library to work with for finding approximate nearest neighbors.

This works with PyTorch v1 pre and fastdotai v1. I am unable to provide support for any of the code.

EDIT: I am also including models after partial refactoring. They can be trained on multiple GPUs, have some issues fixed, etc. For much nicer models with inplace ABN check out the solution I link to above!

Some of the papers with ideas leveraged in the code:
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
* [Squeeze-and-Excitation on Spatial and Temporal Deep Feature Space for Action Recognition](https://arxiv.org/abs/1806.00631)
* [Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752)
* [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323)
* [Bridging Category-level and Instance-level Semantic Image Segmentation](https://arxiv.org/abs/1605.06885)
