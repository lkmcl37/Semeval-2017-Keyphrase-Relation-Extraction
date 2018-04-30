'''
Created on Apr 19, 2018
@author: KaMan Leong
'''
import tensorflow as tf
import numpy as np
import model
import random
import math

def main():
    batch_size = 16
    num_epochs = 2 #epoch size must be <= 2
    save_path = './model/'
    word_emb, train, dev = init_data()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    gpu_config = "/gpu:0"
    
    max_acc = 0.0
    with tf.Session(config=config) as sess:
        with tf.device(gpu_config): 
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                re_model = model.GRU(True, word_emb)
                
            global_step = tf.Variable(0, name="global_step", trainable=False)
            
            train_op = tf.train.AdamOptimizer(0.001).minimize(re_model.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            max_acc = 0.0
            print("Training model...")
            num_iterations = int(math.ceil(1.0 * len(train[0]) / batch_size))
            for epoch in range(num_epochs):
                print("Epoch: ", epoch)
                #shuffle the examples
                sh_index = np.arange(len(train[0]))
                np.random.shuffle(sh_index)
                for i in range(len(train)):
                    train[i] = train[i][sh_index]
                
                for iteration in range(num_iterations):
                    #get a batch
                    word_batch, pos1_batch, pos2_batch, ent_batch, y_batch = get_next_batch(train, iteration * batch_size, batch_size)
                
                    train_shape = []
                    train_word = []
                    train_pos1 = []
                    train_pos2 = []
                    train_ent_type = []
                    #train_partofspeech = []
                    train_word_num = 0
                    
                    #process the batches
                    for i in range(len(word_batch)):
                        train_shape.append(train_word_num)
                        train_word_num += len(word_batch[i])
                        train_word.extend([word for word in word_batch[i]])
                        train_pos1.extend([pos1 for pos1 in pos1_batch[i]])
                        train_pos2.extend([pos2 for pos2 in pos2_batch[i]])
                        train_ent_type.extend([ent for ent in ent_batch[i]])
                        #train_partofspeech.extend([pos for pos in partspeech_batch[i]])
                    
                    train_shape.append(train_word_num)
                    train_shape = np.array(train_shape)
                    train_word = np.array(train_word)
                    train_pos1 = np.array(train_pos1)
                    train_pos2 = np.array(train_pos2)
                    #train_partofspeech = np.array(train_partofspeech)
                    train_ent_type = np.array(train_ent_type)
                    
                    _, step, train_loss, train_acc, _, _ = sess.run(
                        [train_op, global_step, re_model.total_loss, re_model.accuracy, re_model.l2_loss, re_model.final_loss],
                        feed_dict={
                            re_model.input_shape:train_shape,
                            re_model.input_word:train_word,
                            re_model.input_pos1:train_pos1,
                            re_model.input_pos2:train_pos2,
                            re_model.input_ent_type:train_ent_type,
                            #re_model.input_speech:train_partofspeech,
                            re_model.input_y:y_batch
                    })
                    
                    if step %50 == 0:
                        train_acc = np.reshape(np.array(train_acc), (batch_size))
                        train_acc = np.mean(train_acc)
                        print("step {}, loss {:g}, train accuracy {:g}".format(step, train_loss, train_acc))
                        
                    
                    if step%100 == 0:
                        # perform validation
                        dev_order = list(range(len(dev[0])))
                        random_idx = random.randint(0, int(len(dev_order) / float(batch_size)))
                        
                        word_batch, pos1_batch, pos2_batch, ent_batch, dev_y = get_next_batch(dev, random_idx * batch_size, batch_size)
                        dev_shape = []
                        dev_word = []
                        dev_pos1 = []
                        dev_pos2 = []
                        dev_ent_type = []
                        #dev_partofspeech = []
                        dev_word_num = 0
                        
                        for i in range(len(word_batch)):
                            dev_shape.append(dev_word_num)
                            dev_word_num += len(word_batch[i])
                            dev_word.extend([word for word in word_batch[i]])
                            dev_pos1.extend([pos1 for pos1 in pos1_batch[i]])
                            dev_pos2.extend([pos2 for pos2 in pos2_batch[i]])
                            dev_ent_type.extend([ent for ent in ent_batch[i]])
                            #dev_partofspeech.extend([pos for pos in speech_batch[i]])
                         
                        dev_shape.append(dev_word_num)
                        dev_shape = np.array(dev_shape)
                        dev_word = np.array(dev_word)
                        dev_pos1 = np.array(dev_pos1)
                        dev_pos2 = np.array(dev_pos2)
                        #dev_partofspeech = np.array(dev_partofspeech)
                        dev_ent_type = np.array(dev_ent_type)
                    
                        dev_loss, dev_acc = sess.run(
                            [re_model.total_loss, re_model.accuracy],
                            feed_dict={
                                re_model.input_shape:dev_shape,
                                re_model.input_word:dev_word,
                                re_model.input_pos1:dev_pos1,
                                re_model.input_pos2:dev_pos2,
                                re_model.input_ent_type:dev_ent_type,
                                #re_model.input_speech:dev_partofspeech,
                                re_model.input_y:dev_y
                            })
                                            
                        dev_acc = np.reshape(np.array(dev_acc), (batch_size))
                        dev_acc = np.mean(dev_acc)
                        print("dev performance: accuracy {:g}".format(dev_acc))
                        
                        if max_acc < dev_acc:
                            max_acc = dev_acc
                            saver.save(sess, save_path + str(epoch) + '_RE_model.ckpt')
                        
    
#generate a new batch given start index
def get_next_batch(data, start_index, batch_size):
    last_index = start_index + batch_size
    data_len = len(data[0])
    
    temp_word = list(data[0][start_index:min(last_index, data_len)])
    temp_pos1 = list(data[1][start_index:min(last_index, data_len)])
    temp_pos2 = list(data[2][start_index:min(last_index, data_len)])
    
    temp_ent_type = list(data[3][start_index:min(last_index, data_len)])
    #temp_speech_type = list(data[4][start_index:min(last_index, data_len)])
    temp_y = list(data[-1][start_index:min(last_index, data_len)])
                   
    if last_index > len(data[0]):
        left_size = last_index - len(data[0])
        for _ in range(left_size):
            k = np.random.randint(len(data[0]))
            temp_word.append(data[0][k])
            temp_pos1.append(data[1][k])
            temp_pos2.append(data[2][k])
            temp_ent_type.append(data[3][k])
            #temp_speech_type.append(data[4][k])
            temp_y.append(data[-1][k])

    temp_word = np.array(temp_word)
    temp_pos1 = np.array(temp_pos1)
    temp_pos2 = np.array(temp_pos2)
    temp_ent_type = np.array(temp_ent_type)
    #temp_speech_type = np.array(temp_speech_type)
    temp_y = np.array(temp_y)
    
    return temp_word, temp_pos1, temp_pos2, temp_ent_type, temp_y


def init_data():
    print('loading word embedding')
    word_emb = np.load('./data/emb_vec.npy')

    print('reading data')
    train_word = np.load('./vectorized_data/train_word.npy') #token lists
    train_pos1 = np.load('./vectorized_data/train_pos1.npy') #relative position 1
    train_pos2 = np.load('./vectorized_data/train_pos2.npy') #relative position 2
    train_ent = np.load('./vectorized_data/train_type.npy') #keyphrase entity type
    train_part_of_speech = np.load('./vectorized_data/train_partofspeech.npy') #part of speech
    train_y = np.load('./vectorized_data/train_y.npy')
    
    dev_word = np.load('./vectorized_data/dev_word.npy') #same as above, but for the dev set
    dev_pos1 = np.load('./vectorized_data/dev_pos1.npy')
    dev_pos2 = np.load('./vectorized_data/dev_pos2.npy')
    dev_ent = np.load('./vectorized_data/dev_type.npy')
    dev_part_of_speech = np.load('./vectorized_data/dev_partofspeech.npy')
    dev_y = np.load('./vectorized_data/dev_y.npy')

    return word_emb, [train_word, train_pos1, train_pos2, train_ent, train_part_of_speech, train_y], [dev_word, dev_pos1, dev_pos2, dev_ent, dev_part_of_speech, dev_y]

       
if __name__ == "__main__":
    main()
