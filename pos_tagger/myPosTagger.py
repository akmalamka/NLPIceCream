import numpy as np
import pandas as pd
import random
import pprint, time

class POSTagger:
    def __init__(self, train_tagged_words, tags):
        self.train_tagged_words = train_tagged_words
        self.tags = tags
        self.tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
        self.tags_df = []

    # hitung emission probability
    def word_given_tag(self, word, tag):
        tag_list = [pair for pair in self.train_tagged_words if pair[1]==tag]
        count_tag = len(tag_list) # total kemunculan suatu tag di dalam data
        w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
        count_w_given_tag = len(w_given_tag_list) # total kemunculan kata yg dicari dengan tag yang sesuai di dalam data
    
        return (count_w_given_tag, count_tag)

    # hitung transition probability
    def t2_given_t1(self, t2, t1):
        if (t1 == 'S' or t2 == 'E'): # kalau tag yang dicari transisinya ada Start atau End
            tags = [pair[2] for pair in self.train_tagged_words]
            count_t1 = len([t for t in tags if t==t1]) # menghitung kemunculan t1 di dalam data
            count_t2_t1 = 0
            for index in range(len(tags)-1): # menghitung kemunculan t2 setelah t1
                if tags[index]==t1 and tags[index+1] == t2:
                    count_t2_t1 += 1
            return (count_t2_t1, count_t1) # outputnya tuple
        else:
            tags = [pair[1] for pair in self.train_tagged_words]
            count_t1 = len([t for t in tags if t==t1]) # menghitung kemunculan t1 di dalam data
            count_t2_t1 = 0
            for index in range(len(tags)-1): # menghitung kemunculan t2 setelah t1
                if tags[index]==t1 and tags[index+1] == t2:
                    count_t2_t1 += 1
            return (count_t2_t1, count_t1) # outputnya tuple

    def create_tags_df(self):
        for i, t1 in enumerate(list(self.tags)):
            for j, t2 in enumerate(list(self.tags)): 
                if (self.t2_given_t1(t2,t1)[1] == 0) or (t1 == 'S' and t2 == 'S'):
                    self.tags_matrix[i, j] = 0
                else:
                    self.tags_matrix[i, j] = self.t2_given_t1(t2, t1)[0]/self.t2_given_t1(t2, t1)[1]
        self.tags_df = pd.DataFrame(self.tags_matrix, columns = list(self.tags), index=list(self.tags))

    def Viterbi(self, words):
        state = []
        T = list(set([pair[1] for pair in self.train_tagged_words]))
        
        for key, word in enumerate(words):
            #initialise list of probability column for a given observation
            p = [] 
            for tag in T:
                if key == 0:
                    transition_p = self.tags_df.loc['.', tag]
                else:
                    transition_p = self.tags_df.loc[state[-1], tag]
                    
                # compute emission and state probabilities
                emission_p = self.word_given_tag(words[key], tag)[0]/self.word_given_tag(words[key], tag)[1]
                state_probability = emission_p * transition_p    
                p.append(state_probability)
                
            pmax = max(p)
            # getting state for which probability is maximum
            state_max = T[p.index(pmax)] 
            state.append(state_max)
        return list(zip(words, state))

    def test(self, test_set):
        random.seed(1234)      #define a random seed to get same sentences when run multiple times
 
        # choose random 10 numbers
        rndom = [random.randint(1,len(test_set)) for x in range(10)]
        
        # list of 10 sents on which we test the model
        test_run = [test_set[i] for i in rndom]
        for i in range(len(test_run)):
            for j in range(len(test_run[i])):
                test_run[i][j] = (test_run[i][j][0],) + (test_run[i][j][1],)
        # print(test_run)
        # list of tagged words
        test_run_base = [tup for sent in test_run for tup in sent]
        
        # list of untagged words
        test_tagged_words = [tup[0] for sent in test_run for tup in sent]

        start = time.time()
        tagged_seq = self.Viterbi(test_tagged_words)
        end = time.time()
        difference = end-start
        # print(tagged_seq)
        print("Time taken in seconds: ", difference)
        
        # accuracy
        check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
        # print(check)
        accuracy = len(check)/len(tagged_seq)
        print('Viterbi Algorithm Accuracy: ',accuracy*100)


# def main():
#     print("Hello World!")

# if __name__ == "__main__":
#     main()