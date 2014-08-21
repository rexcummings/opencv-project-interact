""" Simple Linear Kalman Filter class. 

This implementation follows the description in 
Computational Principles of Mobile Robotics 2nd Ed., by Dudek and Jenkin.

Author: Nathan Sprague, Rex Cummings 
Version: 1.0, August 21, 2014
"""

import numpy as np


class KalmanFilter(object):
    """ Simple Kalman Filter Class. """

    def __init__(self, Phi, C_v, Lambda, C_w, x_0, P_0):
        """ Construct a Kalman Filter object. 

        Arguments:
           Phi - state transition matrix
           C_v   - covariance matrix of the state transition function
           Lambda - measurement model matrix
           C_w   - measurement covariance
           x_0 - initial state
           P_0 - initial state covariance matrix. 
        """

        self.Phi = Phi
        self.C_v = C_v
        self.Lambda = Lambda
        self.C_w = C_w
        self.x = x_0
        self.P = P_0

    def predict(self):
        """ 
        Perform one step of prediction based on the state transition model. 
        """
        
        # Project the state forward: (Equation 4.12)
        self.x = self.Phi.dot(self.x)
        # Update the state covariance: (Equation 4.13)
        self.P = self.Phi.dot(self.P).dot(self.Phi.T) + self.C_v

    def correct(self, z):        
        """ 
        Update the state estimate with a sensor reading. 

        Arguments:
           z - measurement vector
        """
        
        # Compute the Kalman gain: (Equation 4.15)
        K_top = self.P.dot(self.Lambda.T)
        K_bottom = self.Lambda.dot(self.P).dot(self.Lambda.T) + self.C_w
        K = K_top.dot(np.linalg.inv(K_bottom))
                   
        # Calculate the residual: (Equation 4.16)
        r = z - self.Lambda.dot(self.x)
        
        # Revise the state estimate: (Equation 4.17)
        self.x = self.x + K.dot(r)
        
        # Revise the state covariance: (Equation 4.18)
        self.P = (np.eye(self.P.shape[0]) - K.dot(self.Lambda)).dot(self.P)



if __name__ == "__main__":
    pass
