"""
--------------------------------------------------------------------------
        Class: CS 582: Intro to Speech Processing
      Purpose: In this homework, you will implement the key HMM algorithms in Python
 Instructions:
     1. Write Python code as indicated
     2. Turn in this template with working code for each exercise.
     3. Turn in a sample run to demonstrate your work

        Title: Final Extra Credit
          Due: 5.16.2021
         Name: Mikhail Mineev
 Turn-in Date:
    File Name: Final_Mikhail_Mineev.py
--------------------------------------------------------------------------
"""
from speech_recogn.HMMSpeechRecog import HMMSpeechRecog

# Establish our model
model = HMMSpeechRecog(add_mfcc_delta_delta=False)
model.train(3, 2)
model.test('models/accuracies/accuracy_digit_hmm.txt')

model.pickle('models/digit_hmm.pkl')
model2 = HMMSpeechRecog.unpickle('models/digit_hmm.pkl')

#After the training we simply run the predict on each of the test recordings


# One
predicted_labels1 = model2.predict_files(['data/test_data/one' + '.wav'])
print(f'predicted label {predicted_labels1[0][0]}\n')

# Two
predicted_labels2 = model2.predict_files(['data/test_data/two' + '.wav'])
print(f'predicted label {predicted_labels2[0][0]}\n')

# Three
predicted_labels3 = model2.predict_files(['data/test_data/three' + '.wav'])
print(f'predicted label {predicted_labels3[0][0]}\n')

# Four
predicted_labels4 = model2.predict_files(['data/test_data/four' + '.wav'])
print(f'predicted label {predicted_labels4[0][0]}\n')

# Five
predicted_labels5 = model2.predict_files(['data/test_data/five' + '.wav'])
print(f'predicted label {predicted_labels5[0][0]}\n')

# Six

predicted_labels6 = model2.predict_files(['data/test_data/six' + '.wav'])
print(f'predicted label {predicted_labels6[0][0]}\n')

# Seven
predicted_labels7 = model2.predict_files(['data/test_data/seven' + '.wav'])
print(f'predicted label {predicted_labels7[0][0]}\n')

# Eight
predicted_labels8 = model2.predict_files(['data/test_data/eight' + '.wav'])
print(f'predicted label {predicted_labels8[0][0]}\n')

# Nine
predicted_labels9 = model2.predict_files(['data/test_data/nine' + '.wav'])
print(f'predicted label {predicted_labels9[0][0]}\n')

# Ten
predicted_labels0 = model2.predict_files(['data/test_data/ten' + '.wav'])
print(f'predicted label {predicted_labels0[0][0]}\n')

##model2.calc_mean_entropy()