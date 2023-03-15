import torch
import numpy as np
from scipy import special
## u is mean, v is variance
def approx_1rd_torch(u, v):
    w1 = 1 - torch.erfc(np.sqrt(2)*u/(2*v))/2
    w0 = np.sqrt(2)*v*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi))
    return w0/5, w1
def approx_1rd_loss_torch(u, v):
    loss = -u**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 + np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))*torch.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 - v**2*torch.exp(-u**2/v**2)/(2*np.pi)
    return loss
def approx_2rd_torch(u, v):
    w2 = np.sqrt(2)*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v)
    w1 = -np.sqrt(2)*u*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)*v) - torch.erfc(np.sqrt(2)*u/(2*v))/2 + 1
    w0 = np.sqrt(2)*u**2*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v) + np.sqrt(2)*v*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi))
    return w0/5, w1, w2/10 
def approx_2rd_loss_torch(u, v):
    loss = -u**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 + np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))*torch.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 - 3*v**2*torch.exp(-u**2/v**2)/(4*np.pi)
    return loss
def approx_3rd_torch(u, v):
    w3 = -np.sqrt(2)*u*torch.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**3)
    w2 = np.sqrt(2)*u**2*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v**3) + np.sqrt(2)*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v)
    w1 = -np.sqrt(2)*u**3*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v**3) - np.sqrt(2)*u*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v) - torch.erfc(np.sqrt(2)*u/(2*v))/2 + 1
    w0 = np.sqrt(2)*u**4*torch.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**3) + np.sqrt(2)*v*torch.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi))
    return w0, w1, w2, w3 
def approx_3rd_loss_torch(u, v):
    loss = -u**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 - u**2*torch.exp(-u**2/v**2)/(12*np.pi) + np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))*torch.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 - 3*v**2*torch.exp(-u**2/v**2)/(4*np.pi)
    return loss
def approx_4rd_torch(u, v):
    w4 = np.sqrt(2)*u**2*torch.exp(-u**2/(2*v**2))/(48*np.sqrt(np.pi)*v**5) - np.sqrt(2)*torch.exp(-u**2/(2*v**2))/(48*np.sqrt(np.pi)*v**3)
    w3 = -np.sqrt(2)*u**3*torch.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**5)
    w2 = np.sqrt(2)*u**4*torch.exp(-u**2/(2*v**2))/(8*np.sqrt(np.pi)*v**5) + 3*np.sqrt(2)*torch.exp(-u**2/(2*v**2))/(8*np.sqrt(np.pi)*v)
    w1 = -np.sqrt(2)*u**5*torch.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**5) + np.sqrt(2)*u**3*torch.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**3) - np.sqrt(2)*u*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)*v) - torch.erfc(np.sqrt(2)*u/(2*v))/2 + 1
    w0 = np.sqrt(2)*u**6*torch.exp(-u**2/(2*v**2))/(48*np.sqrt(np.pi)*v**5) - np.sqrt(2)*u**4*torch.exp(-u**2/(2*v**2))/(16*np.sqrt(np.pi)*v**3) + 3*np.sqrt(2)*u**2*torch.exp(-u**2/(2*v**2))/(16*np.sqrt(np.pi)*v) + 3*np.sqrt(2)*v*torch.exp(-u**2/(2*v**2))/(16*np.sqrt(np.pi))
    return w0, w1, w2, w3, w4
def approx_4rd_loss_torch(u, v):
    loss = -u**4*torch.exp(-u**2/v**2)/(48*np.pi*v**2) - u**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 - u**2*torch.exp(-u**2/v**2)/(24*np.pi) + np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))*torch.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*torch.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*torch.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*torch.erfc(np.sqrt(2)*u/(2*v))/2 - 37*v**2*torch.exp(-u**2/v**2)/(48*np.pi)
    return loss