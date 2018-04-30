'''
Created on Apr 18, 2018
@author: KaMan Leong
'''
import tensorflow as tf
 
class GRU:
    def __init__(self, is_training, word_emb):
        self.vocab_size = 21718
        self.sent_len = 111
        self.num_classes = 3
        self.gru_size = 230

        self.num_layers = 1
        self.pos_emb_dim = 10  #position embedding dimension
        self.pos_num = 123
        self.ent_type_num = 7
        self.ent_type_emb_dim = 10  #entity type embedding dimension
        self.batch_size = 16
        self.speech_num = 46        #part of speech tag num
        self.speech_emb_dim = 10    #part of speech embedding dimension
        
        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0
        
        # input tensor
        # word, relative position 1/2, entity type and part-of-speech
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.sent_len], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sent_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sent_len], name='input_pos2')
        self.input_ent_type = tf.placeholder(dtype=tf.int32, shape=[None, self.sent_len], name='type_of_entity')
        #self.input_speech = tf.placeholder(dtype=tf.int32, shape=[None, self.sent_len], name='type_of_speech')
        
        # output tensor
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_y')
        self.input_shape = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='input_shape')
        
        # embeddings
        word_embedding = tf.get_variable(initializer=word_emb, trainable=False, name='word_emb')
        pos1_embedding = tf.get_variable('positon_emb1',[self.pos_num,self.pos_emb_dim])
        pos2_embedding = tf.get_variable('positon_emb2',[self.pos_num,self.pos_emb_dim])
        ent_type_embedding = tf.get_variable('ent_type_emb',[self.ent_type_num,self.ent_type_emb_dim])
       # speech_embedding = tf.get_variable('pos_tag_emb',[self.speech_num,self.speech_emb_dim])
        
        # Bidirectional GRU
        gru_cell_forward = tf.contrib.rnn.GRUCell(self.gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(self.gru_size)

        if is_training:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=0.5)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=0.5)
        
        rnn_cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * self.num_layers)
        rnn_cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * self.num_layers)
        
        # embedding layer
        # word_emb + pos_emb for arg1 + pos_emb for arg2
        rnn_input_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),
                                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2),
                                                   tf.nn.embedding_lookup(ent_type_embedding, self.input_ent_type)])
                                                   #tf.nn.embedding_lookup(speech_embedding, self.input_speech)])
        rnn_input_backward = tf.concat(axis=2,
                                    values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
                                            tf.nn.embedding_lookup(pos2_embedding, tf.reverse(self.input_pos2, [1])),
                                            tf.nn.embedding_lookup(ent_type_embedding, tf.reverse(self.input_ent_type, [1]))])
                                           # tf.nn.embedding_lookup(speech_embedding, tf.reverse(self.input_speech, [1]))])
        
        rnn_output_forward = []
        state_forward = rnn_cell_forward.zero_state(self.input_shape[-1], tf.float32)
        with tf.variable_scope('GRU_forward') as scope:
            for position in range(self.sent_len):
                if position > 0:
                    scope.reuse_variables()
                (rnn_cell_output_forward, state_forward) = rnn_cell_forward(rnn_input_forward[:, position, :], state_forward)
                rnn_output_forward.append(rnn_cell_output_forward)

        rnn_output_backward = []
        state_backward = rnn_cell_backward.zero_state(self.input_shape[-1], tf.float32)
        with tf.variable_scope('GRU_backward') as scope:
            for position in range(self.sent_len):
                if position > 0:
                    scope.reuse_variables()
                (rnn_cell_output_backward, state_backward) = rnn_cell_backward(rnn_input_backward[:, position, :], state_backward)
                rnn_output_backward.append(rnn_cell_output_backward)

        rnn_output_forward = tf.reshape(tf.concat(axis=1, values=rnn_output_forward), [self.input_shape[-1], self.sent_len, self.gru_size])
        rnn_output_backward = tf.reverse(
            tf.reshape(tf.concat(axis=1, values=rnn_output_backward), [self.input_shape[-1], self.sent_len, self.gru_size]),
            [1])
        
        #element-wise sum to combine the forward and backward pass outputs
        output_h = tf.add(rnn_output_forward, rnn_output_backward)
        
        # word-level attention layer
        attention_alpha = tf.get_variable('attention_alpha', [self.gru_size, 1])
        #attension formulas: 
        #M = tahn(H), input vectors
        #alpha = softmax(w*M), r: sentence length, w: dimension of word vectors
        #r = H*alpha, r dimension of word vectors
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [self.input_shape[-1] * self.sent_len, self.gru_size]), attention_alpha),
                       [self.input_shape[-1], self.sent_len])), [self.input_shape[-1], 1, self.sent_len]), output_h), [self.input_shape[-1], self.gru_size])
        
        
        # sentence-level attention layer      
        sent_repre_list = []
        sent_alpha_list = []
        sent_s_list = []
        sent_out = []
        
        sent_alpha = tf.get_variable('sent_attention_alpha',[self.gru_size])
        sent_r = tf.get_variable('sent_attention_r',[self.gru_size,1])        
        
        relation_emb = tf.get_variable('relation_emb',[self.num_classes, self.gru_size])
        sent_d = tf.get_variable('bias_d',[self.num_classes])
        
        #for each pair of related key-phrase:
        #get representation of sentence
        #do sentence-level attention
        #make predication on relation type
        for i in range(self.batch_size):
            sent_repre_list.append(tf.tanh(attention_r[self.input_shape[i]:self.input_shape[i+1]]))
            batch_size = self.input_shape[i+1] - self.input_shape[i]

            sent_alpha_list.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sent_repre_list[i], sent_alpha), sent_r), [batch_size])),
                           [1, batch_size]))
            
            sent_s_list.append(tf.reshape(tf.matmul(sent_alpha_list[i], sent_repre_list[i]), [self.gru_size, 1]))
            sent_out.append(tf.add(tf.reshape(tf.matmul(relation_emb, sent_s_list[i]), [self.num_classes]), sent_d))
            self.prob.append(tf.nn.softmax(sent_out[i]))
            
            self.predictions.append(tf.argmax(self.prob[i], 0, name="preds"))
            
            self.loss.append(
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sent_out[i], labels=self.input_y[i])))
            
            self.total_loss = self.loss[i] if i == 0 else self.total_loss + self.loss[i]
               
            self.accuracy.append(
                tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                               name="acc"))

        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        
