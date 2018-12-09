import matplotlib.pyplot as plt 
import initialize_hmm
import matplotlib.pyplot as plt


def vis_activity_of_one_person(index,y,s):
    # 
    index = int(index)
    my_xticks = ['1: WALKING','2: WALKING_UPSTAIRS','3: WALKING_DOWNSTAIRS','4: SITTING','5: STANDING','6: LAYING']
    plt.plot(range(len(y[s==index])),y[s==index])
    plt.yticks(range(1,7),my_xticks )
    plt.tight_layout()
    plt.show()


def visualize_features(ACT1,ACT2):
    # x: (7352,561)
    # y: (7352,)
    # ACT1 and ACT 2 indicates two sets of activities that you want to compare

    x_train, y_train, s_train, x_test, y_test, s_test = initialize_hmm.load_data()

    # standardize it (get z-scores)
    x_train = initialize_hmm.standardize_data(x_train) 
    # get the indices of each activity sequence 
    segments = initialize_hmm.segment_data(y_train)  

    n_feature = x_train.shape[1]

    # 1 "segment" is "5  from 0   to   27"
    act1_seq = []
    act2_seq = []


    # iterate over all features and draw a plot for each feature
    # for debugging purpose draw the first 3 features.

    for f in range(n_feature):
        plt.figure()
        act1_seq = []
        act2_seq = []
        for i in range(len(segments)): # interate over all segments
            if segments[i,0] in ACT1:
                act1_seq.append(x_train[segments[i,1]:segments[i,2],f].ravel())
            if segments[i,0] in ACT2:
                act2_seq.append(x_train[segments[i,1]:segments[i,2],f].ravel())

        
        alpha = 0.5
        for i, seq in enumerate(act1_seq):
            if i==0:
                plt.plot(range(len(seq)), seq,'r-',label=','.join([str(x) for x in ACT1]),alpha=alpha)
            else:
                plt.plot(range(len(seq)), seq,'r-',alpha=alpha)
        for i, seq in enumerate(act2_seq):
            if i==0:
                plt.plot(range(len(seq)), seq,'b-',label=','.join([str(x) for x in ACT2]),alpha=alpha)
            else:
                plt.plot(range(len(seq)), seq,'b-',alpha=alpha)
        plt.title('Feature {}/{}'.format(f+1,n_feature))
        plt.xlabel('Time Frame')
        plt.ylabel('Feature Value')
        plt.legend()
        # plt.show()
        plt.savefig('feature{}.png'.format(f))

    