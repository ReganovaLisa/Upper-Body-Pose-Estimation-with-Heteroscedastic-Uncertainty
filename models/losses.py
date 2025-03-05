import tensorflow as tf


# def mse_loss_single_output_with_uncertainty(y_true, y_pred):
#     """
#     Mean squared error loss with uncertainty computed for the model that outputs yaw pitch, roll as vectors
#     of two elements: the first the angle, the second the uncertainty associated to it
#
#     Args:
#         :y_true (): two-dimensional vector containing the groundtruth of the angle (the second dimension contains
#             the continuous values, the first the binned values)
#         :y_pred (): the predicted angle of the model
#
#     Returns:
#         :loss (): mse loss computed between the real and predicted angle
#     """
#     uncertainty = y_pred[:, 1]
#     y_pred = y_pred[:, 0]
#
#     cont_true = y_true[:, 1]
#
#     squared_error = tf.math.square(tf.math.abs(cont_true - y_pred))
#     inv_std = tf.math.exp(-uncertainty)
#     mse = tf.reduce_mean(inv_std * squared_error)
#     reg = tf.reduce_mean(uncertainty)
#     loss = 0.5 * (mse + reg)
#
#     return loss

# class Mse_loss_single_output_with_uncertainty(tf.keras.losses.Loss):

#     def call(self, y_true, y_pred):
#         """
#             Mean squared error loss with uncertainty computed for the model that outputs yaw pitch, roll as vectors
#             of two elements: the first the angle, the second the uncertainty associated to it

#             Args:
#                 :y_true (): two-dimensional vector containing the groundtruth of the angle (the second dimension contains
#                     the continuous values, the first the binned values)
#                 :y_pred (): the predicted angle of the model

#             Returns:
#                 :loss (): mse loss computed between the real and predicted angle
#             """

#         # print(y_true.shape) = (None,2)

#         # print(f'y_true.shape = {y_true.shape}')
#         # print(f'y_predicted.shape before= {y_pred.shape}')

#         uncertainty = y_pred[:, 1]
#         y_pred = y_pred[:, 0]

#         # print(f'y_predicted.shape after= {y_pred.shape}')
#         # print(f'predicted value = {y_pred}')

#         cont_true = y_true[:, 0]# cont_true = y_true[:, 1]
#         # print(f'cont_true.shape = {cont_true.shape}')
#         # print(f'true value = {cont_true}')

#         squared_error = tf.math.square(tf.math.abs(cont_true - y_pred))
#         inv_std = tf.math.exp(-uncertainty)
#         mse = tf.reduce_mean(inv_std * squared_error)
#         reg = tf.reduce_mean(uncertainty)
#         print(mse)
#         print(reg)
#         loss = 0.5 * (mse + reg)

#         return loss

class Mse_loss_single_output_with_uncertainty(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """
        Mean squared error loss with uncertainty for angles.

        Args:
            y_true (tf.Tensor): Ground truth values. Shape: (batch_size, 2)
            y_pred (tf.Tensor): Predicted values. Shape: (batch_size, 2)

        Returns:
            tf.Tensor: Computed loss.
        """

        # Ensure y_pred has the correct shape
        tf.debugging.assert_equal(tf.shape(y_pred)[-1], 2, "y_pred must have 2 elements per sample: [angle, uncertainty]")

        # Extract predicted angle and uncertainty
        uncertainty = y_pred[:, 1]
        y_pred_angle = y_pred[:, 0]

        # Extract ground truth angle
        cont_true = y_true[:, 0]

        # Compute squared error
        squared_error = tf.math.square(cont_true - y_pred_angle)

        # Compute inverse standard deviation with clipping for stability
        inv_std = tf.math.exp(-uncertainty)

        # Compute MSE weighted by uncertainty
        mse = tf.reduce_mean(inv_std * squared_error)

        # Regularization term
        reg = tf.reduce_mean(uncertainty)

        # Debugging (optional)
        #tf.print("MSE:", mse, "Regularization:", reg)

        loss = 0.5 * mse + 0.1 * reg

        return loss
    

class Mse_loss_single_output_with_uncertainty_zero_loss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """
        Mean squared error loss with uncertainty for angles.

        Args:
            y_true (tf.Tensor): Ground truth values. Shape: (batch_size, 2)
            y_pred (tf.Tensor): Predicted values. Shape: (batch_size, 2)

        Returns:
            tf.Tensor: Computed loss.
        """

     

        return 0



class Mse_loss_beta_nll(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        """
        Mean squared error loss with uncertainty for angles.

        Args:
            y_true (tf.Tensor): Ground truth values. Shape: (batch_size, 2)
            y_pred (tf.Tensor): Predicted values. Shape: (batch_size, 2)

        Returns:
            tf.Tensor: Computed loss.
        """

        # Ensure y_pred has the correct shape
        tf.debugging.assert_equal(tf.shape(y_pred)[-1], 2, "y_pred must have 2 elements per sample: [angle, uncertainty]")

        # Extract predicted angle and uncertainty
        uncertainty = y_pred[:, 1]
        y_pred_angle = y_pred[:, 0]

        # Extract ground truth angle
        cont_true = y_true[:, 0]

        # Compute squared error
        squared_error = tf.math.square(cont_true - y_pred_angle)

        # Compute inverse standard deviation with clipping for stability
        inv_std = tf.math.exp(-uncertainty)

        # Compute beta term where beta is const between 0 and 1

        beta_coef = 0.5

        beta_term = tf.math.exp(beta_coef * uncertainty)

        # Compute MSE weighted by uncertainty
        mse = tf.reduce_mean(inv_std * squared_error)

        # Regularization term
        reg = tf.reduce_mean(uncertainty)

        # Debugging (optional)
        #tf.print("MSE:", mse, "Regularization:", reg)

        loss = 0.5 * beta_term * (mse +  reg)

        return loss
    


class Mse_loss_beta_nll_adj(tf.keras.losses.Loss):
    def __init__(self, beta_coef=0.5, **kwargs):
        """
        Initialize the loss function with a beta coefficient.

        Args:
            beta_coef (float): Beta coefficient, default is 0.5.
            **kwargs: Additional arguments for the base class.
        """
        super(Mse_loss_beta_nll_adj, self).__init__(**kwargs)
        self.beta_coef = beta_coef

    def call(self, y_true, y_pred):
        """
        Mean squared error loss with uncertainty for angles.

        Args:
            y_true (tf.Tensor): Ground truth values. Shape: (batch_size, 2)
            y_pred (tf.Tensor): Predicted values. Shape: (batch_size, 2)

        Returns:
            tf.Tensor: Computed loss.
        """

        # Ensure y_pred has the correct shape
        tf.debugging.assert_equal(tf.shape(y_pred)[-1], 2, "y_pred must have 2 elements per sample: [angle, uncertainty]")

        # Extract predicted angle and uncertainty
        uncertainty = y_pred[:, 1]
        y_pred_angle = y_pred[:, 0]

        # Extract ground truth angle
        cont_true = y_true[:, 0]

        # Compute squared error
        squared_error = tf.math.square(cont_true - y_pred_angle)

        # Compute inverse standard deviation with clipping for stability
        inv_std = tf.math.exp(-uncertainty)

        # Compute beta term where beta is const between 0 and 1
        beta_term = tf.math.exp(self.beta_coef * uncertainty)

        # Compute MSE weighted by uncertainty
        mse = tf.reduce_mean(inv_std * squared_error)

        # Regularization term
        reg = tf.reduce_mean(uncertainty)

        # Debugging (optional)
        #tf.print("MSE:", mse, "Regularization:", reg)

        loss = 0.5 * beta_term * (mse +  reg)

        return loss

