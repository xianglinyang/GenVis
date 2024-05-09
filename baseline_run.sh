#!/bin/bash

python dvi_main.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR10 1> /home/yangxl21/DVI_data/ResNet_CIFAR10/dvi_log 2>&1
python dvi_main.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR100 1> /home/yangxl21/DVI_data/ResNet_CIFAR100/dvi_log 2>&1
python dvi_main.py --content_path /home/yangxl21/DVI_data/ResNet_FOOD101 1> /home/yangxl21/DVI_data/ResNet_FOOD101/dvi_log 2>&1
python dvi_main.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR10 1> /home/yangxl21/DVI_data/ViT_CIFAR10/dvi_log 2>&1
python dvi_main.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR100 1> /home/yangxl21/DVI_data/ViT_CIFAR100/dvi_log 2>&1
python dvi_main.py --content_path /home/yangxl21/DVI_data/ViT_FOOD101 1> /home/yangxl21/DVI_data/ViT_FOOD101/dvi_log 2>&1

python timevis_main.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR10 1> /home/yangxl21/DVI_data/ResNet_CIFAR10/timevis_log 2>&1
python timevis_main.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR100 1> /home/yangxl21/DVI_data/ResNet_CIFAR100/timevis_log 2>&1
python timevis_main.py --content_path /home/yangxl21/DVI_data/ResNet_FOOD101 1> /home/yangxl21/DVI_data/ResNet_FOOD101/timevis_log 2>&1
python timevis_main.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR10 1> /home/yangxl21/DVI_data/ViT_CIFAR10/timevis_log 2>&1
python timevis_main.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR100 1> /home/yangxl21/DVI_data/ViT_CIFAR100/timevis_log 2>&1
python timevis_main.py --content_path /home/yangxl21/DVI_data/ViT_FOOD101 1> /home/yangxl21/DVI_data/ViT_FOOD101/timevis_log 2>&1

