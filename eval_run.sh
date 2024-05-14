#!/bin/bash

python dvi_test.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR10
python dvi_test.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR100
python dvi_test.py --content_path /home/yangxl21/DVI_data/ResNet_FOOD101
python dvi_test.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR10
python dvi_test.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR100
python dvi_test.py --content_path /home/yangxl21/DVI_data/ViT_FOOD101

python timevis_test.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR10
python timevis_test.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR100
python timevis_test.py --content_path /home/yangxl21/DVI_data/ResNet_FOOD101
python timevis_test.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR10
python timevis_test.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR100
python timevis_test.py --content_path /home/yangxl21/DVI_data/ViT_FOOD101

python tdvi_test.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR10
python tdvi_test.py --content_path /home/yangxl21/DVI_data/ResNet_CIFAR100
python tdvi_test.py --content_path /home/yangxl21/DVI_data/ResNet_FOOD101
python tdvi_test.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR10
python tdvi_test.py --content_path /home/yangxl21/DVI_data/ViT_CIFAR100
python tdvi_test.py --content_path /home/yangxl21/DVI_data/ViT_FOOD101

