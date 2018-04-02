Agenda:
1. How to use this scripts to train new Classifier model.
2. How to make serving after training new model.
3. How to generate all of its necessary library automatically.

""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""" TRAIN CLASSIFIER MODEL """"""""""""
""""""""""""""""""""""""""""""""""""""""""""""""

Step 1: Insert data for training and testing model. All of both files must follow
the defined structures: [<label>\t<data>]. For example, traing file may contains
serveral samples like:

__label__very_nagative  fuck you
__label__postitive  happy
...

Step 2: Training model using train.py script. Please read all the defined arguments
in this script for more detail.

Step 3: After finishing training, trained model (we called old model) can be loaded
and used for predicting by using OldModel class in train_continue.py. We also defined
some useful function for continue training with given new dataset. Please read examples
in this script for more detail

""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""    SERVING SAVED MODEL """"""""""""
""""""""""""""""""""""""""""""""""""""""""""""""

Step 1: Run script at_create_template.py to copy all necessary materials to new defined folder
which is used for serving. Example:

"""
python at_create_template.py --new_path='/home/hoaivan/Desktop/TestHelloCookieCutter1/test' --saved_model_name='sentiment_standard_standford_full_fine_attention_2603'
"""

Note that: <new_path> includes new folder path, <saved_model_name> includes name of folder
containing model.ckpt file, often it appears in saved_model folder.
One important thing: after copying, we should take a look at at_classifier.py and change its
best_model_path variable to path of new saved_model_name.


Step 2: In the new folder, we can do: 1. predict for given sentence, 2. continue training.
For both purposes, code examples were made and stored in at_classifier.py. But, firstly we
need to start server, run:

"""
celery -A at_classifier_server worker --loglevel=info  -Q  at_sentiment -c 2
"""

In this example above, at_sentiment is name of queue in Celery (we should remember it to use
later for client). 2 is the number of concurrence instances.


""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""" GENERATING REQUIREMENT """"""""""""
""""""""""""""""""""""""""""""""""""""""""""""""
Step1: cd to this folder, then run command:
"""
pipreqs .
"""
