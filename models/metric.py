import tensorflow as tf

# class Mean_Absolute_Error_HPE(tf.keras.metrics.Metric):
#     def __init__(self, name="MAE_hpe", **kwargs):
#         super(Mean_Absolute_Error_HPE, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name="ctp", initializer="zeros")
#         self.score = 50.0

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # print(f'y_pred {y_pred}')
#         #
#         # print(f' y_pred shape = {tf.shape(y_pred)} and {tf.rank(y_pred)}')
#         # print(f' y_true shape = {tf.shape(y_true)} and {tf.rank(y_true)}')
#         y_pred = y_pred[:,0] # takes only first coloumn
#         y_true = y_true[:,0]

#         print(f' y_pred shape = {tf.shape(y_pred)}')
#         self.score = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)))

#     def result(self):
#         return self.score

#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.score = 20.0

# # class Mean_Absolute_Error_HPE_ypr(tf.keras.metrics.Metric):
# #     def __init__(self, name="MAE_hpe_ypr", **kwargs):
# #         super(Mean_Absolute_Error_HPE_ypr, self).__init__(name=name, **kwargs)
# #         self.true_positives = self.add_weight(name="ctp", initializer="zeros")
# #         self.score = 50.0
# #
# #     def update_state(self, y_true, y_pred, sample_weight=None):
# #         # print(f'y_pred {y_pred}')
# #         #
# #         # print(f' y_pred shape = {tf.shape(y_pred)} and {tf.rank(y_pred)}')
# #         # print(f' y_true shape = {tf.shape(y_true)} and {tf.rank(y_true)}')
# #         y_pred = y_pred[:,0] # takes only first coloumn
# #         y_true = y_true[:,0]
# #
# #         print(f' y_pred shape = {tf.shape(y_pred)}')
# #         self.score = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)))
# #
# #     def result(self):
# #         return self.score

#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.score = 20.0

class Save_Uncertainty_deg(tf.keras.metrics.Metric):
    def __init__(self, name="Save_UNC", **kwargs):
        super(Save_Uncertainty_deg, self).__init__(name=name, **kwargs)
        self.score = 50.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        # print(f'y_pred {y_pred}')
        #
        # print(f' y_pred shape = {tf.shape(y_pred)} and {tf.rank(y_pred)}')
        # print(f' y_true shape = {tf.shape(y_true)} and {tf.rank(y_true)}')
        unc = y_pred[:,1] # takes only second coloumn

        #TODO check here if correct

        self.score = tf.reduce_mean(tf.sqrt(tf.exp(unc)))
        #this is sigma

    def result(self):
        return self.score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.score = 20.0

# class Angle_accuracy(tf.keras.metrics.Metric):
#     def __init__(self, name="angle_accuracy", **kwargs):
#         super(Angle_accuracy, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name="ctp", initializer="zeros")
#         self.score = 50.0

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # print(f'y_pred {y_pred}')
#         #
#         # print(f' y_pred shape = {tf.shape(y_pred)} and {tf.rank(y_pred)}')
#         # print(f' y_true shape = {tf.shape(y_true)} and {tf.rank(y_true)}')
#         y_pred = y_pred[:,0] # takes only first coloumn
#         y_true = y_true[:,0]

#         print(f' y_pred shape = {tf.shape(y_pred)}')
#         self.score = tf.reduce_mean(tf.cast((tf.where((tf.abs(tf.subtract(y_pred, y_true)) < 15), 1, 0)), tf.float16))
#     def result(self):
#         return self.score

#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.score = 2.0


class Mean_Absolute_Error_HPE(tf.keras.metrics.Metric):
    def __init__(self, name="MAE_hpe", **kwargs):
        super(Mean_Absolute_Error_HPE, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name="total_error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract first column (predicted angle and ground truth angle)
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]

        # Compute absolute error
        error = tf.abs(y_pred - y_true)

        # Update state
        self.total_error.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        # Return mean absolute error
        return self.total_error / self.count

    def reset_states(self):
        # Reset the state variables
        self.total_error.assign(0.0)
        self.count.assign(0.0)


# class Save_Uncertainty_deg(tf.keras.metrics.Metric):
#     def __init__(self, name="Save_UNC", **kwargs):
#         super(Save_Uncertainty_deg, self).__init__(name=name, **kwargs)
#         self.total_uncertainty = self.add_weight(name="total_uncertainty", initializer="zeros")
#         self.count = self.add_weight(name="count", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Extract uncertainty (second column of predictions)
#         unc = y_pred[:, 1]

#         # Compute sqrt(exp(unc)) as sigma
#         sigma = tf.sqrt(tf.exp(unc))

#         # Update state
#         self.total_uncertainty.assign_add(tf.reduce_sum(sigma))
#         self.count.assign_add(tf.cast(tf.shape(unc)[0], tf.float32))

#     def result(self):
#         # Return average uncertainty
#         return self.total_uncertainty / self.count

#     def reset_states(self):
#         # Reset the state variables
#         self.total_uncertainty.assign(0.0)
#         self.count.assign(0.0)

class Angle_accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="angle_accuracy", **kwargs):
        super(Angle_accuracy, self).__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name="correct_predictions", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract first column (predicted angle and ground truth angle)
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]

        # Compute accuracy (absolute difference < 15 degrees)
        correct = tf.cast(tf.abs(y_pred - y_true) < 15, tf.float32)

        # Update state
        self.correct_predictions.assign_add(tf.reduce_sum(correct))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        # Return accuracy
        return self.correct_predictions / self.count

    def reset_states(self):
        # Reset the state variables
        self.correct_predictions.assign(0.0)
        self.count.assign(0.0)
