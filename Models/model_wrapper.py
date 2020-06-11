from abc import ABC, abstractmethod
import tensorflow as tf

class CLInterface(ABC):
    """A general interface for our models, that are used for continual learning."""

    def __init__(self):
        """Initialize the network.
        Args:
        """
        super(CLInterface, self).__init__()

        # The following member variables have to be set by all classes that
        # implement this interface.
        self.in_dim = None
        self.out_dim = None
        self.learning_rate = None
        self.optimizer = None


    def _is_properly_setup(self):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self.in_dim is not None)
        assert(self.out_dim is not None)
        assert(self.learning_rate is not None)
        assert (self.optimizer is not None)

    @property
    def num_outputs(self):
        """Getter for the attribute num_outputs."""
        return self.out_dim

    @property
    def num_inputs(self):
        """Getter for the attribute num_outputs."""
        return self.in_dim

    @property
    def model_name(self):
        pass

    @property
    def name(self):
        pass

    @abstractmethod
    def _get_graph(self):
        pass # TODO implement

    @abstractmethod
    def train_model(self, trainsets, valsets, sess, no_epochs, batch_size, ):
        '''
        Train the model and store the gradient and evaluation result

        :param trainsets:
        :param valsets:
        :param sess:
        :param no_epochs:
        :param batch_size:
        :return:
        '''
        pass # TODO implement

    @abstractmethod
    def _train_step(self):
        pass  # TODO implement

    #@abstractmethod
    def test_model(selfs, testsets):
        '''
        test the current model and return test result
        :param testsets:
        :return:
        '''
        pass # TODO implement

    @abstractmethod
    def _assign_pretrained_ops(self):
        """
        Given the pre-trained weights and the mapping from pytorch variable names to tf variable names, assign the
        weights to the model parameters.
        """
        pass