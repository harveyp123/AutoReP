import torch
def x1act_auto(x_input, para, scale_x1 = 1):
    '''
    Applies the ax + b Unit (x1act) function element-wise:
        x2act(x) = w1*x+w0
    '''
    return para[1] * scale_x1 * x_input + para[0]

def x2act_auto(x_input, para, scale_x2 = 1):
    '''
    Applies the ax + b Unit (x1act) function element-wise:
        x2act(x) = w2*x^2 + w1*x+w0
    '''
    return scale_x2*para[2] * torch.mul(x_input, x_input) + para[1] * x_input + para[0]

# ######## y = x^2 + x ########
# def x2act_auto(x_input, para, scale_x2 = 1):
#     '''
#     Applies the ax + b Unit (x1act) function element-wise:
#         x2act(x) = x^2
#     '''
#     return torch.mul(x_input, x_input) + x_input

# def x2act_auto(x_input, para, scale_x2 = 1):
#     '''
#     Applies the ax + b Unit (x1act) function element-wise:
#         x2act(x) = w2*x^2 + w1*x+w0
#     '''
#     return 0.14 * torch.mul(x_input, x_input) + 0.5 * x_input + 0.28

def x3act_auto(x_input, para):
    '''
    Applies the ax + b Unit (x1act) function element-wise:
        x2act(x) = w3*x^3 + w2*x^2 + w1*x+w0
    '''
    x2 = torch.mul(x_input, x_input)
    return para[3] * torch.mul(x2, x_input) + para[2] * x2 + para[1] * x_input + para[0]

def x4act_auto(x_input, para):
    '''
    Applies the ax + b Unit (x1act) function element-wise:
        x2act(x) = w4*x^4 + w3*x^3 + w2*x^2 + w1*x+w0
    '''
    x2 = torch.mul(x_input, x_input)
    x3 = torch.mul(x2, x_input)
    return para[4] * torch.mul(x3, x_input) + para[3] * x3 + para[2] * x2 + para[1] * x_input + para[0]


# def x1act_auto(x_input, scale_x = 0.5, bias = 0.4):
#     '''
#     Applies the x^2 Unit (x2act) function element-wise:
#         x2act(x) = scale*w0*x^2+w1*x+c
#     '''
#     return x_input

# def x2act_auto(x_input, scale_x2 = 0.2, scale_x = 0.5, bias = 0.2):
#     '''
#     Applies the x^2 Unit (x2act) function element-wise:
#         x2act(x) = scale*w0*x^2+w1*x+c
#     '''
#     return scale_x2 * torch.mul(x_input, x_input) + scale_x * x_input + bias