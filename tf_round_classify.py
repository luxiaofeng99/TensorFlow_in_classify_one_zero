# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:57:49 2019

@author: Luwenming
"""
import tensorflow as tf 
import numpy      as np
import matplotlib.pyplot as plt

np.random.seed(100)                     #随机种子
long=7000
A=(np.random.rand(long,1)*2.2-1.1)*1     #1环的圆圈
B=(np.random.rand(long,1)*2.2-1.1)*1
D=np.power(A,2)+np.power(B,2)
E=np.power(D,0.5)
F=np.zeros((long, 2))           #初始化
#print(A)                       
for i in range(len(A)):
    for j in range(1,2+1):
        if j-1<=E[i]<j:F[i][j-1]=1
        
G = np.hstack((E,F))
D_in = np.hstack((A,B))          
D_out= F                    #做好了

#print(D_in)
for i in range(long):
    if D_out[i][0]==0:
        plt.plot(D_in[i][0],D_in[i][1], 'g^')       #绿色
    else:
        plt.plot(D_in[i][0],D_in[i][1], 'b^')       #蓝色

plt.savefig('test1.png')



data_in  = tf.placeholder(tf.float32,shape=(None,2),name='data_in' )
data_out = tf.placeholder(tf.float32,shape=(None,2),name='data_out')

Hk_1 = tf.Variable(tf.random_normal([2,5],mean=0.2,stddev=1,seed=52))   #5个隐藏单元，可以调试
Hk_2 = tf.Variable(tf.random_normal([5,2],mean=0.2,stddev=1,seed=52))

Hb_1 = tf.Variable(tf.random_normal([1,5],mean=0.2,stddev=1,seed=47))
Hb_2 = tf.Variable(tf.random_normal([1,2],mean=0.2,stddev=1,seed=47))

temp1_1 = tf.nn.relu(tf.add(tf.matmul(data_in,Hk_1),Hb_1))        #隐藏层
last_out= tf.nn.softmax(tf.add(tf.matmul(temp1_1,Hk_2),Hb_2))     #输出层

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_out, logits=last_out))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tall = 50000        #次数可以调试
    for i in range(tall):      
        sess.run(train_step,feed_dict={data_in:D_in,data_out:D_out})
        if i%5000==1:
            total_loss = sess.run(cross_entropy,feed_dict = {data_in:D_in,data_out:D_out})    
            print('已经训练了 %d 轮,loss : %g' % (i,total_loss))
    print('已经训练了 %d 轮,loss : %g' % (tall,total_loss))            
    print(sess.run(Hk_1))
    print(sess.run(Hk_2))
    print(sess.run(Hb_1))
    print(sess.run(Hb_2))
    '''    
[[-3.887658   -0.24080557 -4.5000315   5.5210805   0.12362359]
 [ 1.0899833  -5.4261746  -0.7833127  -0.2927809   5.052207  ]]

[[-3.1870275  3.1298487]
 [-3.093297   4.271304 ]
 [-3.2027445  4.0335116]
 [-5.979999   2.327514 ]
 [-3.7010896  3.878781 ]]

[[-2.157274  -1.9498461 -2.4376132 -2.4922962 -1.6648023]]

[[ 12.493833 -11.90079 ]]
    '''
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(last_out, 1), tf.argmax(D_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #print(sess.run(accuracy, feed_dict={data_in: D_in[1] , data_out: D_out[1]}))
        
    X=7000
    Y=0
    Z =sess.run(last_out,feed_dict = {data_in:D_in[0:X]})
    
    plt.figure()         
    plt.subplot(1, 1, 1)  
    
    for i in range(X):
        color_t=str(Z[i][0])         #灰度值
        plt.plot(D_in[i][0],D_in[i][1],color=color_t,marker = 'D')
        #输出图片，黑色为大于1的，白色是小于1的，灰色的处于边缘，是“机器”认为不好说的，loss值主要都在这里

    plt.savefig('test2.png')
    print(Y)
plt.show()









