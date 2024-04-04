
# Note: This is the result of following tutorial strictly:
# link: https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-create-a-play-generator
# https://www.simplyscripts.com/ TERMINATOR 2
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

path_to_file = tf.keras.utils.get_file('t2.txt', 'http://www.scifiscripts.com/scripts/t2.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = { u:i for i, u in enumerate(vocab) }
idx2char = np.array( vocab )

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

def int_to_txt(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

seq_length = 100
exmples_per_epoch = len(text) // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# spliting input target into two hello to hell and ello 
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_txt(x))
    print("\nOUTPUT")
    print(int_to_txt(y))


BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# building models here 
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model 

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# so we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were.

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

"""
New tutorial starts from here
https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-training-the-model

"""

# Compiling the model
# At this point we can think of our prolem as a classification problem where predicts the probability of each unique letter coming next.
# Okay we are going to compile the model with adam optimizer and the loss function as loss, which we defined above as loss function 
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_perfix = os.path.join(checkpoint_dir, "ckpt_(epoch)")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_perfix,
    save_weights_only = True
)

# Training : Finally we will start training the model
# if this is taking a while go to Runtime > Change Runtime Type and Choose GPU under hardware accelerator

history = model.fit(data, epochs=4, callbacks=[checkpoint_callback])

# Loading the Model
# We'll rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one piece of text to the model and have it make a prediction.

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# Once the model is finished training we can find the latest checkpoint that stores the models weights the following line
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# We can load any checkpoint we want by specifying the exact file to load 
checkpoint_num = 10

# Generating the text
def generate_text(model, start_string):
    # Evaluate step (Generating text using the learned model)
    
    # Number of characters to generate
    num_generate = 800
    
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    [9,8,7]
    input_eval = tf.expand_dims(input_eval, 0)
    
    # empty string to store our results
    text_generated = []
    
    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0
    
    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        # we pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))
        
inp = "Sarah runs after him, her bare feet slapping the cold linoleum. Her hospital gown floats out behind her as she dream-runs along the seemingly infinite corridor.  She reaches the corner, slides around it, and..."
print(generate_text(model, inp))

"""
From Tutorial : 6:50
Now, what I'm going to do is go to my other window here, where I've actually typed all of the code just in full and do a quick
summary of everything that we've done. just because there was a lot that went on. And then from there, I'm actually going to train
this on a B movie script and show you kind of how that works in comparison to the Romeo and Juliet. 

Okay so what I'm in now is just the exact same notebook we had before, but I've just pretty much copied all the text
in here. Or it's the exact code we had before. So we just don't have all that other text in between. So I can kind of do a short 
summary of what we did, as well as show you how this worked when I trained it on the B movie Script.

So I did mention I was going to show you that I'm not lying, I will show you can see I've got B movie.txt loaded in here.
And in fact actually I'm gonna just show you the script first, to show you what it looks like. So this is what the B movie script
looks like. You can see it just like a long you know, SCRIPT of text, I just downloaded this for free off the internet. And it's 
actually not as long as the Romeo and Juilet Play.

So we're not going to get as good of results from our model. But it should hopefully be okay. So,we just start and I'm just gonna do
a brief summary.And then I'll show you the results from the B movie Script, just so that people that are confused and maybe have something 
that wraps it up here, we're doing our imports, I don't think I need to explain that this part up herer is just loading in your file, again

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

I don't think I need to explain that, then we're actually going to read the file. So open it from our directory, decode it into UTF-8, we're
going to create a vocabulary and encode all of the text that's inside of this file , then what we're going to do is turn all of that text. 
Yep, into, you know, the encoded version, we're writing a function here that goes the other way around.

>>>>> def int_to_text():

So from int to text, not from the text to int, we are going to define the sequence length that we want to train with, which will be sequence 
length of 100, you can decrease this value, if you want you go  50 or 20, it doesnot really matter, it's up to you, it just that's going to 
determine how many training examples, you're going to have right as the sequence length. 

Next, what we're going to do is create a character dataset from a tensor slices from text as int, well this is going to do is just convert
our entire text that's now a integer array into a bunch of slices of characters. 

Um, so that's what this is doing here. So are not slices, what am I saying? You just gonna convert like split that entire array into just 
characters. Like that's pretty much what it's doing. And then we're gonna say sequences equals chardataset.batch, which now is going to 
take all those characters and batch them in length of 101

>>>> def split_input_target(chunk):
We're gonna do then it split all of that into the training examples. So like this. right, hello and ello.
We are gonna map this function to sequeunces, which means we're going to apply this emeans def split  to every single sequence and store 
that in dataset then, 

we're going to find the parameters for our initial network like batches and batch sizes, we are going to shuffle the 
dataset and batch that into those 64 training examples. Then what we're going to make the function that builds the model, which I've 
already discussed, we're going to actually build the model starting with a batch size of 64. 

We're going to create our last function, compile the model, set our checkpoints for saving, and then train the model and make sure that we
say checkpoint callback as the callback for the model, which means it's gonna save evey epoch the weights of the model I had computed at 
that epoch. So after we do that, then our models trained. 

So we've trained the model you can see I trained this on 50 epochs for the B movie script, and then we're gonna do is build the model now 
with a batch size of one. So we can pass with one example of it and get a predictino, we're going to load the most recent weights into our
model from the checkpoint directory that we defined above. 

And then what we're going to do is build the model and tell it to expect the shape one, "none" as its initial input. Now, none just means we
don't know what that value is going to be. But we know we 're gonna have one entry. Alright, so now we have this generate text method, or 
function here, which I've already kind of went through how that works.

And then we can see that if I type input string, so we type, you know, input string, let's say Hello and hit enter, we'll watch and we can 
see that the B movie, you know trained model comes up with its output here. 

Now unfortunately, the B movie script does not work as well as Romeo and Juilet. That's just because Romeo and Juilet longer piece of text.
It's much better only if it's formatted a lot of nicer and a lot more predictable. Yeah, you kind of get the idea
here. And it's kind of cool to see how this performs on different data.

So I would highly recommend that you guys find some training data that you could give this other than just the
Romeo and Juilet or maybe even try another play or something and see what you can get out of it,Also quick side note to 

"To make your model better increase the amount of epochs here ", ideally, you want this loss to be as low as possible, you can see mine was still
actually moving down at epoch 50, you will reach a point where the amount of epochs won't make a difference. Although, with model like this,
more epochs typically the better because it's difficult for it to kind of overfit.

Because all you want it to do really is just kind of learn how the language works, and then be able to replicate that to you almost right.
So that's kind of the idea here. And with that being said, I'm gonna say that this section is probably done. 

>>>> Now, I know this was a long, probably confusing section for a lot of you and myself. But this is you know what happens when you start getting
into some more complex things in machine learning, it's very difficult to kind of grasp and understand all these concepts in an hour of me just 
explaining them. What I try to do in these videos is introduce to the syntax show you how to get a working, you

know, kind of prototype and hopefully gain enough knowledge the fact where if you're confused by something that I said,
you can go and you can look that up and you can figure our kind of the more important details for yourself because I really 
just I can't go into all you know, the extremes, in these videos. So always that has been this section I hope you guys enjoyed doing this I thought 
this was pretty cool.

>>>>>> And in the next section, we're going to be talking about reinforcement learning. 

"""
