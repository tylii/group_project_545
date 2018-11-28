import matplotlib.pyplot as plt 

def vis_activity_of_one_person(index,y,s):
    # 
    index = int(index)
    my_xticks = ['1: WALKING','2: WALKING_UPSTAIRS','3: WALKING_DOWNSTAIRS','4: SITTING','5: STANDING','6: LAYING']
    plt.plot(range(len(y[s==index])),y[s==index])
    plt.yticks(range(1,7),my_xticks )
    plt.tight_layout()
    plt.show()
