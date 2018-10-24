import pandas as pd
import deepcut as dc
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf


class osh_preprocess:
    def __init__(self):

        pd.set_option('display.max_rows', 2000)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 10000)

    ## 1-Export a model and vocab
    def local_word2vec(self):

        # 1
        df = pd.read_excel('balanced_query_label_all.xlsx')

        punct = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{}~'  # `|` is not present here

        transtab = str.maketrans(dict.fromkeys(punct, ''))

        df['query'] = '|'.join(df['query'].tolist()).translate(transtab).split('|')

        # Standardize English word - transform all word in table to lower case
        df['query'] = df['query'].str.lower()

        query_only = df['query'].values


        # Use deepcut to tokenize Thai word
        # Add Custom_dict ******

        queries = []
        counter = 0

        for i in query_only:
            list_token = dc.tokenize(i, custom_dict=['ไม้ฝา', 'เฌอร่า','ไม้']) # ******************************
            query = list(filter((' ').__ne__, list_token))

            queries.append(query)
            print (counter+1 , ' of ', len(query_only))
            counter+=1

        # maybe classify into two groups, type1,2
        model = Word2Vec(queries, window=1, size=100, min_count=1)
        model.save("1_word2vec_model/query_window1.model")

    ## 2-use model to tranform text into vector
    def vectorizer(self):

        df = pd.read_excel('balanced_query_label_all.xlsx')

        punct = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{}~'  # `|` is not present here

        transtab = str.maketrans(dict.fromkeys(punct, ''))

        df['query'] = '|'.join(df['query'].tolist()).translate(transtab).split('|')

        # Standardize English word - transform all word in table to lower case
        df['query'] = df['query'].str.lower()

        queries = df['query'].values

        model = Word2Vec.load("1_word2vec_model/query_window1.model")

        list_vec = []
        counter = 0

        for query in queries:


            list_text = dc.tokenize(query, custom_dict=['ไม้ฝา', 'เฌอร่า', 'ไม้'])  # **********************
            query = list(filter((' ').__ne__, list_text))

            vec = np.zeros((1, 100))

            for text in query:
                vec += model.wv.word_vec(text)

            vec_reshape = vec.ravel()

            list_vec.append(vec_reshape)

            #print(vec, ' word of: ', query)
            print(counter + 1, ' of ', len(queries))
            counter += 1


        # a = np.asarray(list_vec)
        np.savetxt("2_encoded_query/vectorized.csv", list_vec, delimiter=",") # **********************

        print('## 2-vectorize process is Done ')

        ## train new words *
        '''

        try:
            while True:

                text = input('this input: ').lower()
                vec = np.zeros((1,100))

                ## if condition train new word that is not in the list
                if text in model.wv.index2word:

                    similar = model.similar_by_word(text, topn=10)
                    vec += model.wv.word_vec(text)

                else:

                    some_list = []

                    list_token = dc.tokenize(text, custom_dict=['ไม้ฝา', 'เฌอร่า', 'ไม้']) #********** custom dict
                    query = list(filter((' ').__ne__, list_token))
                    some_list.append(query)

                    print ('some error')
                    model.train(some_list, total_examples= len(some_list), epochs=model.iter)
                    model.build_vocab(some_list, update=True)
                    # model.save('word2vec_query.model')

                    similar = model.wv.similar_by_word(text, topn=10)

                    print ('find new word!! and train done!!')

                print (similar)

        except KeyboardInterrupt:
            pass

        '''

        '''
        ## add word vector to query vector
        vec = np.zeros((1,100))
        x = ['ปูน', 'งาน']
        for i in x:

            vec += model.wv.word_vec(i)

        scaled_inputs = preprocessing.scale(vec)
        print ('scale input: ', scaled_inputs)
        print (vec)
        # shape (1,100)
        print (vec.shape)


        '''


    ## Additional
    def random_data(self):

        df = pd.read_csv('2_encoded_query/vectorized.csv')

        #print (df.info())

        rand = df.sample(frac= 1)
        #print (rand.info())
        #print (rand)

        df2 = pd.DataFrame(rand)
        df2.to_csv('2_encoded_query/rand_vectorized.csv', encoding='utf-8')

    ## 3-To transform csv to npz format in this method
    def npz_pre_processing(self):


        raw_data = np.loadtxt('2_encoded_query/rand_vectorized.csv', delimiter=',')
        inputs_all = raw_data[:, 1:-1]

        targets_all = raw_data[:, -1]

        #scaled_inputs = preprocessing.scale(inputs_all)


        samples_count = inputs_all.shape[0]


        # 80 percentage of train dataset
        train_samples_count = int(0.8 * samples_count)
        # 10 per of validation
        validation_samples_count = int(0.1 * samples_count)

        test_samples_count = samples_count - (train_samples_count + validation_samples_count)

        train_inputs = inputs_all[:train_samples_count]
        train_targets = targets_all[:train_samples_count]

        validation_inputs = inputs_all[train_samples_count:train_samples_count + validation_samples_count]
        validation_targets = targets_all[train_samples_count:train_samples_count + validation_samples_count]

        test_inputs = inputs_all[train_samples_count + validation_samples_count:]
        test_targets = targets_all[train_samples_count + validation_samples_count:]

        np.savez('3_npz/vectorized_nocolumn_train', inputs=train_inputs, targets=train_targets)
        np.savez('3_npz/vectorized_nocolumn_validation', inputs=validation_inputs, targets=validation_targets)
        np.savez('3_npz/vectorized_nocolumn_test', inputs=test_inputs, targets=test_targets)

        print('NPZ pre processing is ===> Done')

    ## 4-export model just one file
    def to_graph_def(self):


        saver = tf.train.import_meta_graph('nn_model/DNN_model.meta', clear_devices=True)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        sess = tf.Session()
        saver.restore(sess, "nn_model/DNN_model")

        output_node_names = "y_pred"
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session
            input_graph_def,  # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")
        )

        # Path
        output_graph = "nn_model/frozen_test_model.pb"
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        sess.close()


    def to_test_model_from_graph(self):

        model = Word2Vec.load("1_word2vec_model/query_window1.model")


        try:
            while True:

                text = input('To test model loading, Fill your query: ').lower()

                if text == '':
                    print ('Your query is Empty!!!, Please fill it again')
                    pass
                else:

                #print('---->', text)

                # **********
                    input_node = 100
                    token_text = dc.tokenize(text, custom_dict=['ไม้ฝา', 'เฌอร่า', 'ไม้'])
                    query = list(filter((' ').__ne__, token_text))


                    print(query)

                    # ทำ manual vectorizer
                    vec = np.zeros((1, 100))

                    for i in query:

                        if i in model.wv.index2word:

                            vec += model.wv.word_vec(i)
                        else:
                            pass
                            print (i , ' :not found in dic')


                    x_batch = vec.reshape(1, input_node)

                    print ('x_batch: ', x_batch)
                    # WARNING !!!!

                    frozen_graph = "nn_model/frozen_test_model.pb"


                    with tf.gfile.GFile(frozen_graph, "rb") as f:
                        print('Model loading ...')
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())

                    with tf.Graph().as_default() as graph:
                        tf.import_graph_def(graph_def,
                                            input_map=None,
                                            return_elements=None,
                                            name=""
                                            )
                    ## NOW the complete graph with values has been restored
                    y_pred = graph.get_tensor_by_name("y_pred:0")  # ********************************
                    # y_true = graph.get_tensor_by_name('y_true:0')

                    ## Let's feed the images to the input placeholders
                    x = graph.get_tensor_by_name("x:0")  # *************************
                    sess = tf.Session(graph=graph)

                    ### Creating the feed_dict that is required to be fed to calculate y_pred
                    feed_dict_testing = {x: x_batch}
                    result = sess.run(fetches=y_pred, feed_dict=feed_dict_testing)

                    # 'Probability in term of (type): '
                    print('Probabilities in term of (type): ', result)
                    sess.close()

        except KeyboardInterrupt:
            pass


osh = osh_preprocess()

osh.to_test_model_from_graph()