import os
from main import getAnswers

answers = ['A', 'B', 'C', 'D', 'E','A', 'B', 'C', 'D', 'E']

def getMarks():
    img_path = os.path.join(os.getcwd(), 'img', 'test.jpg')
    marked = getAnswers(img_path)
    marks = 0
    for i in range(10):
        if marked[i] == answers[i]:
            print(f"Question {i+1}: Correct")
            marks+=1
        elif marked[i] == 'N':
            print(f"Question {i+1}: Negative Marking!!!")
            marks-=0.5
        elif marked[i] == 'N/A':
            print(f"Question {i+1}: Not Attempted")
        else:
            print(f"Question {i+1}: Incorrect")

    print(f"Total Marks: {marks}/10")


getMarks()
