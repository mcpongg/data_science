import numpy as np
import tensorflow as tf


class osh_data_reader():

    def __init__(self, dataset, batch_size=None):

        # The dataset that loads is one of "train", "validation", "test".
        # e.g. if I call this class with x('train',5), it will load 'Audiobooks_data_train.npz' with a batch size of 5.

        npz = np.load('3_npz/vectorized_nocolumn_{0}.npz'.format(dataset))

        # Two variables that take the values of the inputs and the targets. Inputs are floats, targets are integers
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

        # Counts the batch number, given the size you feed it later
        # If the batch size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0

        # **********************
        self.batch_count = self.inputs.shape[0] // self.batch_size

    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        # You slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1

        # One-hot encode the targets. In this example it's a bit superfluous since we have a 0/1 column
        # as a target already but we're giving you the code regardless, as it will be useful for any
        # classification task with more than one target column
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1

        # The function will return the inputs batch and the one-hot encoded targets
        return inputs_batch, targets_one_hot

    # A method needed for iterating over the batches, as we will put them in a loop
    # This tells Python that the class we're defining is iterable, i.e. that we can use it like:
    # for input, output in data:
    # do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate through its objects
    def __iter__(self):
        return self


# 33 is number of each columns / Don't count index and label column /18/ 24
input_size = 100
# 2 is assigned label or class
output_size = 2
# Hyper parameter /50 /30 for new
hidden_layer_size = 10

tf.reset_default_graph()
# ---------------------------------- START ASSIGN LAYERS --------------------------------

inputs = tf.placeholder(tf.float32, [None, input_size], name='x')
targets = tf.placeholder(tf.float32, [None, output_size], name='y_true')

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])

# Activation function +++++++++++++++
# relu: REctified Linear Unit
outputs_1 = tf.nn.leaky_relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])

outputs_2 = tf.nn.leaky_relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable("biases_3", [hidden_layer_size])

outputs_3 = tf.nn.sigmoid(tf.matmul(outputs_2, weights_3) + biases_3)

weights_4 = tf.get_variable("weights_4", [hidden_layer_size, output_size])
biases_4 = tf.get_variable("biases_4", [output_size])

outputs = tf.matmul(outputs_3, weights_4) + biases_4

# ---------------------------------- END ASSIGN LAYERS --------------------------------

y_pred = tf.nn.sigmoid(outputs, name='y_pred')

# Softmax compute propability of the outputs ( 0-1 )
# softmax_cross_entropy_with_logits_v2
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=targets)

# Find cost value: Difference of predict and actual value.
mean_loss = tf.reduce_mean(loss)

# Optimizer is hyperparameter +++ 0.001
optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mean_loss)

# argmax is finding the maximum weight of output in one-hot format
out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))

accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

# ---------------------------------- START TENSORFLOW SESSION ----------------------------------

sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()

sess.run(initializer)

# 100
batch_size = 10

max_epochs = 300

prev_validation_loss = 9999999.

# Train and validation part ----------------------------------
train_data = osh_data_reader('train', batch_size)
validation_data = osh_data_reader('validation')

whole_thing = []

train_loss_list = []
iteration_list = []
val_loss_list = []
val_acc_list = []

# Save NN model
saver = tf.train.Saver()

for epoch_counter in range(max_epochs):

    curr_epoch_loss = 0.

    # Start training data set
    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimize, mean_loss], feed_dict={inputs: input_batch, targets: target_batch})
        curr_epoch_loss += batch_loss

    curr_epoch_loss /= train_data.batch_count

    validation_loss = 0.
    validation_accuracy = 0.

    for input_batch, target_batch in validation_data:
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                                                        feed_dict={inputs: input_batch, targets: target_batch})
    '''
    whole_thing.append((epoch_counter+1, validation_loss, validation_accuracy, curr_epoch_loss))

    iteration_list.append(epoch_counter+1)
    val_loss_list.append(validation_loss)
    val_acc_list.append(validation_accuracy)
    train_loss_list.append(curr_epoch_loss)
    '''

    print('Epoch ' + str(epoch_counter + 1) +
          '. Training loss: ' + '{0:.3f}'.format(curr_epoch_loss) +
          '. Validation loss: ' + '{0:.3f}'.format(validation_loss) +
          '. Validation accuracy: ' + '{0:.2f}'.format(validation_accuracy * 100.) + '%')
    '''

    '''
    if (epoch_counter + 1) % 10 == 0:
        print('         Epoch ' + str(epoch_counter + 1) +
              '. Training loss: ' + '{0:.3f}'.format(curr_epoch_loss) +
              '. Validation loss: ' + '{0:.3f}'.format(validation_loss) +
              '. Validation accuracy: ' + '{0:.2f}'.format(validation_accuracy * 100.) + '%')
    '''
    if validation_loss > prev_validation_loss:
        break
    '''

    prev_validation_loss = validation_loss

print('End of training.')

'''
df = pd.DataFrame(whole_thing, columns=['epochs', 'validation_loss', 'validation_accuracy', 'train_loss'])
df.set_index(df.epochs, drop=True, inplace=True)
df.to_excel('ANN_result.xlsx', encoding='utf-8')
'''

# Test part ----------------------------------
test_data = osh_data_reader('test')

for input_batch, target_batch in test_data:
    test_accuracy = sess.run([accuracy], feed_dict={inputs: input_batch, targets: target_batch})

test_acc_percent = test_accuracy[0] * 100
print('Test accuracy: ' + '{0:.2f}'.format(test_acc_percent) + '%')

save_path = saver.save(sess, 'nn_model/DNN_model')
print("Model saved in path: %s" % save_path)

sess.close()

# ---------------------------------- END TENSORFLOW SESSION ----------------------------------
