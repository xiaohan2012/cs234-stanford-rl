import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule

from configs.q2_linear import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        ##############################################################
        """
        TODO: 
            Add placeholders:
            Remember that we stack 4 consecutive frames together.
                - self.s: batch of states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.a: batch of actions, type = int32
                    shape = (batch_size)
                - self.r: batch of rewards, type = float32
                    shape = (batch_size)
n                - self.sp: batch of next states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.done_mask: batch of done, type = bool
                    shape = (batch_size)
                - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: 
            Variables from config are accessible with self.config.variable_name.
            Check the use of None in the dimension for tensorflow placeholders.
            You can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################

        remain_dims = tuple(state_shape[:-1]) + (state_shape[-1] * self.config.state_history, )
        self.s = tf.placeholder(tf.float32, shape=(None, ) + remain_dims)
        self.a = tf.placeholder(tf.int32, shape=(None, ))
        self.r = tf.placeholder(tf.float32, shape=(None, ))
        self.sp = tf.placeholder(tf.float32, shape=(None, ) + remain_dims)
        self.done_mask = tf.placeholder(tf.bool, shape=(None, ))  # termination of episode or not
        self.lr = tf.placeholder(tf.float32)
        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            Implement a fully connected with no hidden layer (linear
            approximation with bias) using tensorflow.

        HINT: 
            - You may find the following functions useful:
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        with tf.variable_scope(scope):
           out = tf.layers.dense(
               tf.layers.flatten(state),
               num_actions, activation=None, use_bias=True,
               reuse=reuse
           )
        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: 
            Add an operator self.update_target_op that for each variable in
           tf.GraphKeys.GLOBAL_VARIABLES that is in q_scope, assigns its
            value to the corresponding variable in target_q_scope

        HINT: 
            You may find the following functions useful:
                - tf.get_collection
                - tf.assign
                - tf.group (the * operator can be used to unpack a list)

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        vars_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)

        vars = sorted(vars, key=lambda v: v.name)
        vars_target = sorted(vars_target, key=lambda v: v.name)

        assert len(vars) == len(vars_target)

        assign_ops = [tf.assign(vt, v) for v, vt in zip(vars, vars_target)]
        self.update_target_op = tf.group(*assign_ops)

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        HINT: 
            - Config variables are accessible through self.config
            - You can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
            - You may find the following functions useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############

        # get q value from target_q
        undone_mask = tf.cast(tf.logical_not(self.done_mask), dtype=tf.float32)
        Q_samp = (
            self.r
            + undone_mask * self.config.gamma * tf.reduce_max(target_q, axis=1)
        )
        
        # get q value from up-to-date estimator
        mask = tf.one_hot(
            self.a,
            depth=num_actions
        )
        Q_s_a = tf.boolean_mask(q, mask)

        self.loss = tf.reduce_mean(
            tf.squared_difference(Q_samp, Q_s_a)
        )
        self.target_q_value = Q_samp
        self.q_value = Q_s_a
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """

        ##############################################################
        """
        TODO: 
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val 
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store 
                this scalar in self.grad_norm

        HINT: you may find the following functions useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variables by writing self.config.variable_name
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############
        opt = tf.train.AdamOptimizer(
            learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07
        )

        with tf.variable_scope(scope):
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            gvs = opt.compute_gradients(self.loss, variables)  # A list of (gradient, variable) pairs.
            self.grad = gvs
            # print('grads', grads)
            if self.config.grad_clip:
                # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
                gvs = [(tf.clip_by_norm(grad, self.config.clip_val), var)
                       for grad, var in gvs]

            self.train_op = opt.apply_gradients(gvs)
            self.grad_norm = tf.global_norm(gvs)

        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
