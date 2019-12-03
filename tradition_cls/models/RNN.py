import torch
import torch.nn as nn

from ccf.tradition_cls.models.Datamodel import DataModel
class RNN(nn.Module):
	def __init__(self,opt):
		super(RNN, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		self.opt = opt
		self.data_model = DataModel(opt)
		self.batch_size = self.opt.batch_size
		self.output_size = self.opt.output_size
		self.hidden_size = self.opt.hidden_size
		self.num_layers = self.opt.num_layers

		self.embedding_length = self.data_model.word_dim

		self.rnn = nn.RNN(self.embedding_length, self.hidden_size, num_layers=self.num_layers, bidirectional=True)
		self.label = nn.Linear(2*self.hidden_size, self.output_size)
	
	def forward(self, input_sentences, batch_size=None):
		
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the final_hidden_state of RNN.
		logits.size() = (batch_size, output_size)
		
		"""
		with torch.no_grad():
			input = self.data_model.get_data(input_sentences)
			input = input.permute(1, 0, 2)
		input = input.cuda()
		'''
		if self.opt.use_gpu:
			if batch_size is None:
				h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # 4 = num_layers*num_directions
			else:
				h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())
		else:
			if batch_size is None:
				h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size)) # 4 = num_layers*num_directions
			else:
				h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size))
		'''
		output, h_n = self.rnn(input)
		# h_n.size() = (4, batch_size, hidden_size)
		h_n = h_n.permute(1, 0, 2) # h_n.size() = (batch_size, 4, hidden_size)
		h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
		# h_n.size() = (batch_size, 4*hidden_size)
		logits = self.label(h_n) # logits.size() = (batch_size, output_size)
		
		return logits
