'''
Created on Apr 19, 2018
@author: KaMan Leong
'''
import math
import tensorflow as tf
import numpy as np
import time
import os
import model

model_path = "./model/best_RE_model.ckpt"
word_emb = np.load('./data/emb_vec.npy')
batch_size = 16
    
#load the vectorized data and annotations from files
def load_file(name):
    
    path = "./vectorized_data/vectorized_test/" +name
    word_path = path + "_word.npy"
    pos1_path = path + "_pos1.npy"
    pos2_path = path + "_pos2.npy"
    type_path = path + "_type.npy"
    #speech_path = path + "_partofspeech.npy"
    y_path = path + "_y.npy"
    ann_path = "./data/test/" + name + ".ann"
    pair_path = path + ".pair"
    
    test_word = np.load(word_path)
    test_pos1 = np.load(pos1_path)
    test_pos2 = np.load(pos2_path)
    test_ent_type = np.load(type_path)
    #test_speech = np.load(speech_path)
    test_y = np.load(y_path)
    
    with open(ann_path, "r", encoding='utf-8') as f:
        ann = f.readlines()
    
    pairs = []
    with open(pair_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            pairs.append(line.split("\t"))
    
    return [test_word, test_pos1, test_pos2, test_ent_type, test_y], ann, pairs
         
         
#output predication results to .ann files for computing precision, recall and F1  
def build_output_file(total_samples, pred, pairs, ann, name, relation2id=["NA", "Hyponym-of", "Synonym-of"]):
    directory = "./test_pred/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    keyphrase_mapper = {}
    output_list = []
    for line in ann:
        line = line.strip()
        if line.startswith("T"):
            fields = line.split()
            word = "_".join(fields[4:]).replace(",", "")
            keyphrase_mapper[word] = fields[0]
            output_list.append(line)
       
    #arrange the results into ann format
    rel = set()
    num_hyponym = 1
    for i, label in enumerate(pred[:total_samples]):
        if pairs[i][0] == pairs[i][1] or keyphrase_mapper[pairs[i][0]] == keyphrase_mapper[pairs[i][1]] or relation2id[int(label)] == "NA":
            continue
        if relation2id[int(label)] == "Hyponym-of":
            outline = "R" + str(num_hyponym) + "\t" + "Hyponym-of" + " " + "Arg1:" + keyphrase_mapper[pairs[i][0]] + " " +"Arg2:" + keyphrase_mapper[pairs[i][1]]
            num_hyponym += 1
        else:
            outline = "*" + "\t" + "Synonym-of" + " " + keyphrase_mapper[pairs[i][0]] + " " + keyphrase_mapper[pairs[i][1]]
        rel.add(outline)
            
    #write results
    output_path = directory + name + ".ann"
    output_list.extend(list(rel))
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_list:
            f.write(line + "\n")
    

def main():
    with tf.Session() as sess:
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            re_model = model.GRU(False, word_emb)
            
            print("loading model parameter...")
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
    
            print("Testing...")
            #get file name/path lists from a txt file
            with open("./vectorized_data/vectorized_test/file_list.txt", "r") as f:
                files = f.readlines()
                for name in files:
                    name = name.strip()
                    print("Predicting file: " + name)
                    if not name:
                        continue
                    test, test_ann, test_pairs = load_file(name)
                    test_order = list(range(len(test[0])))
                    num_iterations = int(math.ceil(1.0 * len(test_order) / batch_size))
                    y_hat = [] #predication
                    
                    for i in range(num_iterations): 
                        word_batch, pos1_batch, pos2_batch, ent_type_batch, y_batch = get_next_batch(test, i * batch_size, batch_size)
                        test_shape = []
                        test_word = []
                        test_pos1 = []
                        test_pos2 = []
                        #test_speech = []
                        test_ent_type = []
                        test_word_num = 0
                        
                        for i in range(len(word_batch)):
                            test_shape.append(test_word_num)
                            test_word_num += len(word_batch[i])
                            test_word.extend([word for word in word_batch[i]])
                            test_pos1.extend([pos1 for pos1 in pos1_batch[i]])
                            test_pos2.extend([pos2 for pos2 in pos2_batch[i]])
                            #test_speech.extend([pos for pos in speech_batch[i]])
                            test_ent_type.extend([ent for ent in ent_type_batch[i]])
                         
                        test_shape.append(test_word_num)
                        test_shape = np.array(test_shape)
                        test_word = np.array(test_word)
                        test_pos1 = np.array(test_pos1)
                        test_pos2 = np.array(test_pos2)
                        #test_speech = np.array(test_speech)
                        test_ent_type = np.array(test_ent_type)
                        
                        pred = sess.run([
                                        re_model.predictions],
                                        feed_dict={
                                            re_model.input_shape:test_shape,
                                            re_model.input_word:test_word,
                                            re_model.input_pos1:test_pos1,
                                            re_model.input_pos2:test_pos2,
                                            re_model.input_ent_type:test_ent_type,
                                            #re_model.input_speech:test_speech,
                                            re_model.input_y:y_batch
                                        })
    
                        y_hat += list(pred[0])
                    
                    #output results to files
                    build_output_file(len(test[0]), y_hat, test_pairs, test_ann, name)

#generate batches for testing
def get_next_batch(data, start_index, batch_size):
    last_index = start_index + batch_size
    data_len = len(data[0])
    
    temp_word = list(data[0][start_index:min(last_index, data_len)])
    temp_pos1 = list(data[1][start_index:min(last_index, data_len)])
    temp_pos2 = list(data[2][start_index:min(last_index, data_len)])
    temp_ent = list(data[3][start_index:min(last_index, data_len)])
    #temp_speech = list(data[4][start_index:min(last_index, data_len)])
    temp_y = list(data[-1][start_index:min(last_index, data_len)])
        
    #if the generated data size is less than batch size, pad more examples
    if last_index > len(data[0]):
        left_size = last_index - len(data[0])
        for _ in range(left_size):
            k = np.random.randint(len(data[0]))
            temp_word.append(data[0][k])
            temp_pos1.append(data[1][k])
            temp_pos2.append(data[2][k])
            temp_ent.append(data[3][k])
            #temp_speech.append(data[4][k])
            temp_y.append(data[-1][k])

    temp_word = np.array(temp_word)
    temp_pos1 = np.array(temp_pos1)
    temp_pos2 = np.array(temp_pos2)
    temp_ent = np.array(temp_ent)
   # temp_speech = np.array(temp_speech)
    temp_y = np.array(temp_y)
    
    return temp_word, temp_pos1, temp_pos2, temp_ent, temp_y

if __name__ == "__main__":
    main()