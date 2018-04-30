'''
Created on Apr 18, 2018
@author: KaMan Leong
'''
import os
import re
import nltk
import spacy
import random
import itertools
import numpy as np
from nltk import word_tokenize
from random import shuffle
nlp = spacy.load('en')

#vectorize the trian data
def read_train_files(inpath, upsampling=False):
    print("Parsing files and generating related pairs")
    file_relation = {}
    
    #extract all sentences containing keyphrase pairs of some relations
    for root, _, files in os.walk(inpath, topdown=True):
        for name in files:
            if not name.endswith(".ann"):
                continue
            file_id = name.split('.')[0]
            dic = []
            token_map = {}
            relation = []
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    fields = line.split('\t')
                    if not line.startswith("T"):
                        if line.startswith("*"):
                            subfields = fields[1].split()
                            mentions = subfields[1:]
                            combination = list(itertools.combinations(mentions, 2))
                            for comb in combination:
                                relation.append([subfields[0], tuple(comb)])
                        elif line.startswith("R"):
                            subfields = fields[1].split()
                            ele = tuple([subfields[1].split(":")[1], subfields[-1].split(":")[1]])
                            relation.append([subfields[0], ele])
                        continue
                    token_map[fields[0]] = [fields[-1], fields[1].split()[0]]

            rel_tuples = set()
            for rel in relation:
                arg1 = rel[1][0]
                arg2 = rel[1][1]

                dic.append([(token_map[arg1][0],token_map[arg2][0]), rel[0], token_map[arg1][-1]])
                if rel[0] == "Syponym-of":
                    rel_tuples.add(tuple([arg2, arg1]))
                rel_tuples.add(tuple([arg1, arg2]))
            
            combination = list(itertools.combinations(token_map.keys(), 2))
            for comb in combination:
                arg1 = comb[0]
                arg2 = comb[1]
                
                arg1_type = token_map[arg1][-1]
                arg2_type = token_map[arg2][-1]
                if (arg1, arg2) not in rel_tuples and arg1 != arg2 and arg1_type == arg2_type:
                    dic.append([(token_map[arg1][0],token_map[arg2][0]), "NA", arg1_type])          
            
            file_relation[file_id] = dic
        
    pos_hy_instances = []
    pos_syn_instances = []
    neg_instances = []
    sent_set = set()
    for root, _, files in os.walk(inpath, topdown=True):
        for name in files:
            if not name.endswith(".txt"):
                continue
            file_id = name.split('.')[0]
            relations = file_relation[file_id]
            
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                doc = nlp(lines[0].strip())
                for rel in relations:
                    for sent in doc.sents:
                        text = sent.text
                        if text.find(rel[0][0]) != -1 and text.find(rel[0][1]) != -1:
                            text = re.sub("[\[].*?[\]]", "", text)
                            
                            temp_rel_1 = '_'.join(rel[0][0].split()).replace(",", "")
                            temp_rel_2 = '_'.join(rel[0][1].split()).replace(",", "")
                            
                            text = text.replace(rel[0][0], temp_rel_1)
                            text = text.replace(rel[0][1], temp_rel_2)
                                
                            text = re.sub("\s\s+", " ", text)
                            rel_list = [temp_rel_1, temp_rel_2, rel[1], rel[-1], text]
                            if rel[1] == "NA":
                                if rel_list[-1] not in sent_set:
                                    neg_instances.append(rel_list)
                                    sent_set.add(rel_list[-1])
                            elif rel[1].startswith("H"):
                                pos_hy_instances.append(rel_list)
                            else:
                                pos_syn_instances.append(rel_list)
                        
        
        instances = []   
        
        #up-sampling technique to balance positive and negative classes         
        if upsampling:
            supposed_size = len(neg_instances)
            shuffle(neg_instances)
            pos_hy_instances = pos_hy_instances*int(supposed_size/len(pos_hy_instances) + 1)
            pos_syn_instances = pos_syn_instances*int(supposed_size/len(pos_syn_instances) + 1)
        
        instances = pos_hy_instances + pos_syn_instances + neg_instances
        random.shuffle(instances)
        
        print("Generated instances: ", len(instances))
        return instances
    

#vectorize the test files
def read_test_files(inpath):
    directory = "./vectorized_data/vectorized_test/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    word2id = build_word2id()
    relation2id = read_relation_mapper()
    type2id = read_keyphrase_type_mapper()
    pos2id = build_pos2id()
    
    print("Generating test data")
    file_list = []
    for root, _, files in os.walk(inpath, topdown=True):
        for name in files:
            dic = []
            token_map = {}
            relation = []
        
            if not name.endswith(".ann"):
                continue
            file_list.append(name.split('.')[0])
            
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    fields = line.split('\t')
                    if not line.startswith("T"):
                        if line.startswith("*"):
                            subfields = fields[1].split()
                            mentions = subfields[1:]
                            combination = list(itertools.combinations(mentions, 2))
                            for comb in combination:
                                relation.append([subfields[0], tuple(comb)])
                        elif line.startswith("R"):
                            subfields = fields[1].split()
                            ele = tuple([subfields[1].split(":")[1], subfields[-1].split(":")[1]])
                            relation.append([subfields[0], ele])
                        continue
                    token_map[fields[0]] = [fields[-1], fields[1].split()[0]]

            rel_tuples = set()
            for rel in relation:
                arg1 = rel[1][0]
                arg2 = rel[1][1]
                dic.append([(token_map[arg1][0],token_map[arg2][0]), rel[0], token_map[arg1][-1]])
                rel_tuples.add(tuple([arg1, arg2]))
            
            combination = list(itertools.combinations(token_map.keys(), 2))
            for comb in combination:
                arg1 = comb[0]
                arg2 = comb[1]
                
                arg1_type = token_map[arg1][-1]
                arg2_type = token_map[arg2][-1]
                if (arg1, arg2) not in rel_tuples and arg1 != arg2 and arg1_type == arg2_type:
                    dic.append([(token_map[arg1][0],token_map[arg2][0]), "NA", arg1_type])        
          
            instances = []
            sent_set = set()
            with open(os.path.join(root, name.split('.')[0] + ".txt"), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                doc = nlp(lines[0].strip())
                for rel in dic:
                    for sent in doc.sents:
                        text = sent.text
                        if text.find(rel[0][0]) != -1 and text.find(rel[0][1]) != -1:
                            text = re.sub("[\[].*?[\]]", "", text)
                            
                            temp_rel_1 = '_'.join(rel[0][0].split()).replace(",", "")
                            temp_rel_2 = '_'.join(rel[0][1].split()).replace(",", "")
                           
                            text = text.replace(rel[0][0], temp_rel_1)
                            text = text.replace(rel[0][1], temp_rel_2)
                             
                            text = re.sub("\s\s+", " ", text)
                            rel_list = [temp_rel_1, temp_rel_2, rel[1], rel[-1], text]
                            if rel_list[2] == "NA":
                                if rel_list[-1] not in sent_set:
                                    instances.append(rel_list)
                                    sent_set.add(rel_list[-1])
                            else:
                                instances.append(rel_list)
                                
                                
            with open(directory + name.split('.')[0] + ".pair", "w", encoding="utf-8") as f:
                for line in instances:
                    f.write(line[0] + "\t" + line[1] + "\n")
                     
            build_vectors(instances, directory + name.split('.')[0], word2id, relation2id, type2id, pos2id)
            with open(directory + "file_list.txt", "w") as f:
                for name in file_list:
                    f.write(name + "\n")
                          
            
#get the dictionary that maps relation to ids
def read_relation_mapper():
    print('Reading relation mapper')
    relation2id = {}
    with open('./data/relation2id.txt', 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            if line == '':
                break
            fields = line.strip().split()
            relation2id[fields[0]] = int(fields[1])
    return relation2id


#helper function for obtaining pre-trained embedding
def getEmbedding(infile_path, data):
    print("building embedding...")
    dimension = 100
    emb = {}
    lexicon = set(data)

    with open(infile_path, "r", encoding="utf-8") as infile:
        for row in infile:
            row = row.strip()            
            items = row.split()
            word = items[0]
            emb[word] = np.array([float(val) for val in items[1:]])

    emb["UNK"] = np.random.normal(size=dimension, loc=0, scale=0.05)
    emb["BLANK"] = np.random.normal(size=dimension, loc=0, scale=0.05)

    vec = []
    with open("./data/emb_vec.txt", "w", encoding="utf-8") as f:
        line = "UNK" + " " + str(emb["UNK"]).replace("\n", "").replace("[", "").replace("]", "")
        f.write(line + "\n")
        
        line = "BLANK" + " " + str(emb["BLANK"]).replace("\n", "").replace("[", "").replace("]", "")
        f.write(line + "\n")

        vec.append(emb["UNK"])
        vec.append(emb["BLANK"])
        for phrase in lexicon:
            for word in phrase.split("_"):
                if word.lower() in emb:
                    word_vec = emb[word.lower()]
                    vec.append(word_vec)
                    line = word + " " + str(word_vec).replace("\n", "").replace("[", "").replace("]", "")
                    f.write(line + "\n")
            
    vec = np.array(vec, dtype=np.float32)
    np.save('./vectorized_data/emb_vec.npy', vec)


#read dictionary that maps entity types to ids
def read_keyphrase_type_mapper():
    print('Reading keyphrase type mapper')
    type2id = {}
    with open('./data/keyphrase_type2id.txt', 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            if line == '':
                break
            fields = line.strip().split()
            type2id[fields[0]] = int(fields[1])
    return type2id


#get dicionary that map tokens to ids
def build_word2id():
    word2id = {}
    with open('./data/emb_vec.txt', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            fields = line.strip().split()
            if line == '':
                break
            word2id[fields[0]] = len(word2id)
    
    return word2id

#map part of speech to id
def build_pos2id():
    pos2id = {}
    with open('./data/pos2id.txt', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            field = line.strip()
            if field:
                pos2id[field] = len(pos2id)
    
    return pos2id
    
    
#helper function to vectorize data
def build_vectors(data, output_file_name, word2id, relation2id, type2id, pos2id):
    #position embedding
    def pos_embed(x):
        if x < -60:
            return 0
        if -60 <= x <= 60:
            return x + 61
        return 122
        
    #helper function to find the index of a token in a sentence
    def find_index(x,y):
        for i in range(len(y)):
            if x == y[i]:
                return i
        return -1
        
    train_sent = {}  
    train_label = {}  # {keyphrase pair:[label1,label2,...]} the label is one-hot vector
    print('Vectorizing the data')
    for fields in data:
        relation_id = relation2id[fields[2]]
        tup = (fields[0], fields[1])  
        label_tag = 0
        if tup not in train_sent:
            train_sent[tup] = []
            train_sent[tup].append([])
            label = [0 for i in range(len(relation2id))]
            label[relation_id] = 1
            train_label[tup] = []
            train_label[tup].append(label)
        else:
            label = [0 for i in range(len(relation2id))]
            label[relation_id] = 1
            temp = find_index(label, train_label[tup])
            if temp == -1:
                train_label[tup].append(label)
                label_tag = len(train_label[tup]) - 1
                train_sent[tup].append([])
            else:
                label_tag = temp

        sentence = word_tokenize(fields[-1])
        #word position
        arg1pos = 0
        arg2pos = 0
        for i in range(len(sentence)):
            if sentence[i] == fields[0]:
                arg1pos = i
            if sentence[i] == fields[1]:
                arg2pos = i
        
        #Embeding the position
        pos1 = []
        pos2 = []
        fixlen = 111
        keyphrase_type = ["O"]*fixlen
        phrase_type = fields[-2] #PROCESS, TASK, MATERIAL
        
        #build position embedding for arg1
        for i in range(0, arg1pos):
            pos1.append(pos_embed(i - arg1pos))
        for i in range(len(fields[0].split("_"))):
            pos1.append(pos_embed(0))
            if i == 0:
                keyphrase_type[i] = "B-" + phrase_type
            else:
                keyphrase_type[i] = "I-" + phrase_type
        for i in range(fixlen - len(pos1)):
            pos1.append(pos_embed(i - arg1pos))
        
        #build position embedding for arg2
        for i in range(0, arg2pos):
            pos2.append(pos_embed(i - arg2pos))
        for i in range(len(fields[1].split("_"))):
            pos2.append(pos_embed(0))
            if i == 0:
                keyphrase_type[i] = "B-" + phrase_type
            else:
                keyphrase_type[i] = "I-" + phrase_type
        for i in range(fixlen - len(pos2)):
            pos2.append(pos_embed(i - arg2pos))
        
        #combine all the feature together
        output = []
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos1[i]
            rel_e2 = pos2[i]
            type = type2id[keyphrase_type[i]]
            output.append([word, rel_e1, rel_e2, type, len(pos2id)])
        
        text = fields[-1].replace(fields[0], fields[0].replace("_", " "))
        text = text.replace(fields[1], fields[1].replace("_", " "))
        sentence = word_tokenize(text)
        #part_of_speech = nltk.pos_tag(text)
        
        for i in range(min(fixlen, len(sentence))):
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word
            #output[i][-1] = pos2id[part_of_speech[i][1]]
        train_sent[tup][label_tag].append(output)
    
    train_x = []
    train_y = []
    with open(output_file_name + '.pair', 'w', encoding='utf-8') as f:
        for i in train_sent:
            lenth = len(train_label[i])
            for j in range(lenth):
                #train_sent[i][j]: idx of each word in sentences
                train_x.append(train_sent[i][j])
                train_y.append(train_label[i][j])
                #arg1, arg2, relation idx
                #the jth arg pairs of ith sentence
                f.write(i[0] + '\t' + i[1] + '\n')
        
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_word = []
    train_pos1 = []
    train_pos2 = []
    train_type = []
    #train_part_of_speech = []

    print("Outputting data vectors")
    for i in range(len(train_x)):
        word = []
        pos1 = []
        pos2 = []
        type = []
       # part_of_speech = []
        for j in train_x[i]:
            word.append([k[0] for k in j])
            pos1.append([k[1] for k in j])
            pos2.append([k[2] for k in j])
            type.append([k[3] for k in j])
            #part_of_speech.append([k[4] for k in j])
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)
        train_type.append(type)
        #train_part_of_speech.append(part_of_speech)
        
    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    train_type = np.array(train_type)
    #train_part_of_speech = np.array(train_part_of_speech)
        
    np.save(output_file_name + '_word.npy', train_word) #tokens
    np.save(output_file_name + '_pos1.npy', train_pos1) #relative position 1
    np.save(output_file_name + '_pos2.npy', train_pos2) #relative position 2
    np.save(output_file_name + '_y.npy', train_y)  #relation tag/class
    np.save(output_file_name + '_type.npy', train_type) #entity_type
    #np.save(output_file_name + '_partofspeech.npy', train_part_of_speech) #part of speech


if __name__ == "__main__":
    print("The vectorization process may take a long time...")
    word2id = build_word2id()
    relation2id = read_relation_mapper()
    type2id = read_keyphrase_type_mapper()
    pos2id = build_pos2id()
    
    data = read_train_files("./data/train/", True)
    build_vectors(data, "./vectorized_data/train", word2id, relation2id, type2id, pos2id)
    
    data = read_train_files("./data/dev/")
    build_vectors(data, "./vectorized_data/dev", word2id, relation2id, type2id, pos2id)
    
    read_test_files("./data/test/")
    