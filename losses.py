import numpy as np 
import torch.nn.functional as F
import torch
import sys 


def boundary_weighted_loss(loss_function, output, target, boundaries_add_factor=None, patch_size=(7,7,7), just_boundary=False, out_boundary_factor=None):
    """    
    Params:
    loss_function: instantiated class of the loss function
    output: logit tensor output of model
    target: the labels of the same shape as output
    pos_weight: tensor of factor to reweight positive samples
    scale: bool on whether to scale output values to a percentage of the largest
    boundaries_add_factor: additional percentage to weight the boundaries
    out_boundary_factor: if used, weighs outside non-placental boundary by this factor
    patch_size: kernel size for average pooling
    just_boundary: True if only want additive boundary loss, False for total scaled loss

    Return: mean voxelwise loss

    """

    epsilon = sys.float_info.epsilon                                         

    loss = loss_function(output, target)
    
    #weight boundaries using probability mask
    if boundaries_add_factor != None and boundaries_add_factor!=0: 
        
        #reshaping output 
        if len(np.shape(output)) == 4:
            new_shape = (1,) + output.shape
            reshaped_output = torch.reshape(target, new_shape)
        else:
            reshaped_output = target

        #calculate appropriate padding based on kernel size
        padding = int((patch_size[0]-1)/2)

        #3d avg pooling to make boundary mask
        output_avg_pool = F.avg_pool3d(reshaped_output, kernel_size= patch_size, stride =(1,1,1), padding=padding, count_include_pad = False) 
        reshaped_boundaries = (output_avg_pool > epsilon) & (output_avg_pool < 1-epsilon)
        reshaped_boundaries_int = reshaped_boundaries.int()                            

        if out_boundary_factor != None: 
            #determine outside and inside boundary  
            outside_boundary = reshaped_boundaries_int.float() - target == 1.0
            outside_boundary = outside_boundary.int()
            inside_boundary = reshaped_boundaries_int.float() - outside_boundary.float() == 1.0
            inside_boundary = inside_boundary.int()

            #weight both boundary masks and apply
            weighted_inner_boundary = (inside_boundary.float() * boundaries_add_factor) 
            weighted_outer_boundary = (outside_boundary.float() * out_boundary_factor)
            weighted_boundaries = weighted_inner_boundary + weighted_outer_boundary
            final_weighted_boundaries = torch.reshape(weighted_boundaries, target.shape)
        else: 
            #weight boundary mask and apply 
            weighted_boundaries = reshaped_boundaries_int.float() * boundaries_add_factor
            final_weighted_boundaries = torch.reshape(weighted_boundaries, target.shape)
        
        additive_loss = (loss * final_weighted_boundaries)
        if just_boundary:
            loss = additive_loss
        else: 
            loss += additive_loss

    loss.type(torch.FloatTensor)
    mloss = torch.mean(loss)

    return mloss