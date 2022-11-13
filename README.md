# Code for paper: Class Inteference of Deep Networks


## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Commands
```
# Start training with: 
python main.py

# Generating class ego directions:
python model_gradient_classwise.py --lr 0.0001 --lr_mode constant --model vgg19

# Generate the loss data in the interference space
python ego_models.py --lr 0.0001 --lr_mode constant --model vgg19 --resolution high --c1 cat --c2 dog

# Visualize using the generated loss data
python vis_f_in_cat_dog_ego_directions.py --lr 0.0001 --lr_mode constant --model vgg19 --c1 cat --c2 dog


# Detect class interference in training
python main_class_loss.py

# Plot the Class Dance and Dancing Notes:
python vis_fc_in_ego_directions.py --c cat --c1 cat --c2 dog
```

## Paper:
[Class Inteference of Deep Networks](https://arxiv.org/abs/2211.01370). Dongcui Diao, Hengshuai Yao, Bei Jiang. 2022. 
