import math
import numpy as np
from scipy import special
## u is mean, v is variance
def approx_1rd(u, v):
    w1 = 1 - special.erfc(np.sqrt(2)*u/(2*v))/2
    w0 = np.sqrt(2)*v*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi))
    return w0/10, w1
def approx_1rd_loss(u, v):
    loss = -u**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*special.erfc(np.sqrt(2)*u/(2*v))/2 + np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))*special.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*special.erfc(np.sqrt(2)*u/(2*v))/2 - v**2*np.exp(-u**2/v**2)/(2*np.pi)
    return loss
def approx_2rd(u, v):
    w2 = np.sqrt(2)*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v)
    w1 = -np.sqrt(2)*u*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)*v) - special.erfc(np.sqrt(2)*u/(2*v))/2 + 1
    w0 = np.sqrt(2)*u**2*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v) + np.sqrt(2)*v*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi))
    return w0/10, w1, w2/10 
def approx_2rd_loss(u, v):
    loss = -u**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*special.erfc(np.sqrt(2)*u/(2*v))/2 + np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))*special.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*special.erfc(np.sqrt(2)*u/(2*v))/2 - 3*v**2*np.exp(-u**2/v**2)/(4*np.pi)
    return loss
def approx_3rd(u, v):
    w3 = -np.sqrt(2)*u*np.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**3)
    w2 = np.sqrt(2)*u**2*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v**3) + np.sqrt(2)*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v)
    w1 = -np.sqrt(2)*u**3*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v**3) - np.sqrt(2)*u*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi)*v) - special.erfc(np.sqrt(2)*u/(2*v))/2 + 1
    w0 = np.sqrt(2)*u**4*np.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**3) + np.sqrt(2)*v*np.exp(-u**2/(2*v**2))/(4*np.sqrt(np.pi))
    return w0, w1, w2, w3 
def approx_3rd_loss(u, v):
    loss = -u**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*special.erfc(np.sqrt(2)*u/(2*v))/2 - u**2*np.exp(-u**2/v**2)/(12*np.pi) + np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))*special.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*special.erfc(np.sqrt(2)*u/(2*v))/2 - 3*v**2*np.exp(-u**2/v**2)/(4*np.pi)
    return loss
def approx_4rd(u, v):
    w4 = np.sqrt(2)*u**2*np.exp(-u**2/(2*v**2))/(48*np.sqrt(np.pi)*v**5) - np.sqrt(2)*np.exp(-u**2/(2*v**2))/(48*np.sqrt(np.pi)*v**3)
    w3 = -np.sqrt(2)*u**3*np.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**5)
    w2 = np.sqrt(2)*u**4*np.exp(-u**2/(2*v**2))/(8*np.sqrt(np.pi)*v**5) + 3*np.sqrt(2)*np.exp(-u**2/(2*v**2))/(8*np.sqrt(np.pi)*v)
    w1 = -np.sqrt(2)*u**5*np.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**5) + np.sqrt(2)*u**3*np.exp(-u**2/(2*v**2))/(12*np.sqrt(np.pi)*v**3) - np.sqrt(2)*u*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)*v) - special.erfc(np.sqrt(2)*u/(2*v))/2 + 1
    w0 = np.sqrt(2)*u**6*np.exp(-u**2/(2*v**2))/(48*np.sqrt(np.pi)*v**5) - np.sqrt(2)*u**4*np.exp(-u**2/(2*v**2))/(16*np.sqrt(np.pi)*v**3) + 3*np.sqrt(2)*u**2*np.exp(-u**2/(2*v**2))/(16*np.sqrt(np.pi)*v) + 3*np.sqrt(2)*v*np.exp(-u**2/(2*v**2))/(16*np.sqrt(np.pi))
    return w0, w1, w2, w3, w4
def approx_4rd_loss(u, v):
    loss = -u**4*np.exp(-u**2/v**2)/(48*np.pi*v**2) - u**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + u**2*special.erfc(np.sqrt(2)*u/(2*v))/2 - u**2*np.exp(-u**2/v**2)/(24*np.pi) + np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))*special.erfc(np.sqrt(2)*u/(2*v))/(2*np.sqrt(np.pi)) - np.sqrt(2)*u*v*np.exp(-u**2/(2*v**2))/(2*np.sqrt(np.pi)) - v**2*special.erfc(np.sqrt(2)*u/(2*v))**2/4 + v**2*special.erfc(np.sqrt(2)*u/(2*v))/2 - 37*v**2*np.exp(-u**2/v**2)/(48*np.pi)
    return loss