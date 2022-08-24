python main.py --lr 0.0001 --lr_mode constant --model vgg19
python model_gradient_classwise.py --lr 0.0001 --lr_mode constant --model vgg19
python ego_models.py --lr 0.0001 --lr_mode constant --model vgg19 --resolution high --c1 cat --c2 dog
