# Code for paper: Class Inteference of Deep Networks


## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Commands

1. Train a model and save the final model. For example, 
>`python main.py --model ResNet18 --lr 0.1 --lr_mode schedule` 

trains a ResNet18 model with SGD aided with learning rate annealing, starting with 0.1 rate and Cosine annealing. 

2. Generating class ego directions: 
>`python model_gradient_classwise.py --lr 0.0001 --lr_mode constant --model vgg19`

3. Generate the loss data in the interference space. 
>`python ego_models.py --lr 0.0001 --lr_mode constant --model vgg19 --resolution high --c1 cat --c2 dog`

4. Visualize using the generated loss data. 
>`python vis_f_in_cat_dog_ego_directions.py --lr 0.0001 --lr_mode constant --model vgg19 --c1 cat --c2 dog`

5. Detect class interference in training. 
>`python main_class_loss.py`

6. Plot the Class Dance and Dancing Notes. 
>`python vis_fc_in_ego_directions.py --c cat --c1 cat --c2 dog`

## Paper:
[Class Inteference of Deep Networks](https://arxiv.org/abs/2211.01370). Dongcui Diao, Hengshuai Yao, Bei Jiang. 2022. 
