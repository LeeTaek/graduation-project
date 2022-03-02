
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn

tf.set_random_seed(777)


# In[2]:


# 파일에서 읽어서 2차원 리스트로
# 곡을 M, L, K, Q, Chords 순서의 리스트로
# 다시 리스트로 저장

tempo = input("박자를 입력하시오.")
key = input("키를 입력하시오.")



with open(r'SampleJigsMusicSheet.csv',) as f :
    a = f.read().split('I')    

song = list()

for i in range(len(a)):
    b = a[i].split(',')
    c = list()
    
    for j in range(len(b)):
        c.append(b[j])
    
    
    # 박자 입력
    if(c[1] == tempo):
    
        # 코드상에 콤마(,)가 있을 경우
        if(len(c) > 6): 
            for k in range(len(c)- 6):
                c[5] = c[5] + ',' + c[6]
                del c[6]
        
        song.append(c)

print(len(song))        
print(song)
    


# In[3]:


# 코드 문자셋 
# 코드에서 사용한 문자열를 하나하나 번호를 붙혀
# 딕셔너리로 관리
chord_set = []

for i in range(len(song)-1):
    chord_set.extend(list(set(song[i+1][5])))

chord_dic = {w: i for i, w in enumerate(chord_set)}

print(chord_set)
print(chord_dic)


# In[6]:


# RNN 함수에 들어갈 매개변수 생성

data_dim = len(chord_set)
hidden_size = len(chord_set)
num_classes = len(chord_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1


# In[7]:


# 학습 시킬 데이터의 입력과 출력 리스트 선언

dataX = []
dataY = []

# 그래프 노드
with tf.name_scope("Wx") as scope:
    y = tf
    


# In[8]:


# 1마디(문자 10개) 기준으로 선언한 리스트에 데이터 입력


for j in range(len(song)-1):
    for i in range(0, len(song[j+1][5]) - sequence_length):
        x_str = song[j+1][5][i:i + sequence_length]
        y_str = song[j+1][5][i+1 : i + sequence_length + 1]
        print(i, x_str, '->', y_str)
    
        x = [chord_dic[c] for c in x_str]
        y = [chord_dic[c] for c in y_str]
    
        dataX.append(x)
        dataY.append(y)


# In[9]:


# 입력시킬 한번에 dataX만큼 학습시킴(batch_size)

batch_size = len(dataX)
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

print(batch_size)


# In[10]:


#one hot 생성

X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)


# In[11]:


# lstm 함수

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell


# In[12]:


# 셀 선언
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)


# In[13]:


outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)


# In[14]:


#
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)


# In[15]:


outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])


# In[16]:


weights = tf.ones([batch_size, sequence_length])


# In[17]:


sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets= Y, weights=weights)


# In[18]:


mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)


# In[19]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[20]:


# 학습 
for i in range(2000):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([chord_set[t] for t in index]), l)



# In[21]:


# 결과 출력 


print("M:" + tempo)
print("L:1/8")
print("K:" + key)

results = sess.run(outputs, feed_dict={X: dataX})

for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([chord_set[t] for t in index]), end='')
    else:
        print(chord_set[index[-1]], end='')

