# Code for paper: Class Inteference of Deep Networks


## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Commands

1. Train a model and save the final model. For example, to train a ResNet18 model with SGD aided with learning rate annealing, starting with 0.1 rate and Cosine annealing. 
>`python main.py --model ResNet18 --lr 0.1 --lr_mode schedule` 

2. Plot CCTM: Cross-Class Test accuracy Matrix/Map. 
   1. Run `python generate_cctm.py` This is used to generate the data for Figure 1 and Table 1 in the paper. 
   2. After this, run `python plot_cctm.py` to generate Figure 1. 

3. Generating class ego directions. The saved ego directions will be used to generate the interference models. 
>`python model_gradient_classwise.py --lr 0.0001 --lr_mode constant --model vgg19`

4. Generate the loss data in the interference space. Note it has two loss modes. 
   1. `python ego_models.py --lr 0.0001 --lr_mode constant --model vgg19 --resolution high --c1 cat --c2 dog --loss gross` generates the overall loss conditioned on the ego space (c1, c2). 
   2. `python ego_models.py --lr 0.0001 --lr_mode constant --model vgg19 --resolution high --c1 cat --c2 dog --loss class-wise` generates the class-wise loss for each class in this ego space. 

5. Visualize the overall loss in the ego space of (class c1, class c2) for model VGG19. This is for  Fig 2/3/4(overall test loss)  
>`python plot_overall_loss_in_ego_space.py -model vgg19 -c1 cat -c2 dog`

6. Visualize the loss for a class $c$ in the ego space of (class c1, class c2) for model VGG19. This is for Fig 5.  
>`python plot_class_loss_in_ego_space.py -model vgg19 -c cat -c1 cat -c2 dog`

7. Plot Fig 6 (inteference dancing) and Fig 7 (interference notes), using
> `python plot_interference_dancing_and_notes.py` Before this, you need to generate the data first, using 
>`python main_class_loss.py`


## Paper:
[Class Inteference of Deep Networks](https://arxiv.org/abs/2211.01370). Dongcui Diao, Hengshuai Yao, Bei Jiang. 2022. 
