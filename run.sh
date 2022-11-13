python main.py --lr 0.0001 --lr_mode constant --model vgg19
python model_gradient_classwise.py --lr 0.0001 --lr_mode constant --model vgg19
python ego_models.py --lr 0.0001 --lr_mode constant --model vgg19 --resolution high --c1 cat --c2 dog
python vis_f_in_cat_dog_ego_directions.py --lr 0.0001 --lr_mode constant --model vgg19 --c1 cat --c2 dog
python ego_models.py --lr 0.1 --lr_mode schedule --model vgg19 --loss classwise --limit_theta 1. --resolution low
