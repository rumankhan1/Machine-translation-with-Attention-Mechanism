<h1>Machine Translation with Global Attention Mechanism</h1>
<p>This notebook shows how to implement Global Attention ( aka Bahdanau Attention ) for Sequence to sequence Machine Translation. We'll make a model to translate from English to Spanish.</p>
<br>
<p><b>Please note:</b></p>
<br>
<ol>
  <li>To better understand from a notebook, you should understand Sequence Sequence models and why we even have this attention mechanism. </li>
  <li>This is a first-cut solution. You can improve it by training on more extensive data.</li>
</ol>
<br>
<h2>About Dataset</h2>
<p>http://www.manythings.org/anki/ is the best source for the data to train a machine translation model. I took Spanish - English dataset. The dataset contains language translation pairs.</p>

<p>After downloading it, we'll perform the following on the data set:</p>
<ul>
  <li>Initially, data is in a text file, so we'll load the data to the Pandas Dataframe</li>
  <li>We'll apply basic processing like strip, removing numeric, etc</li>
  <li>We'll add '<start>' at the beginning of each sentence and '<end>' at each sentence's end.</li>
</ul>
<br>

<h2>Prepare Train and Validation Set</h2>
<p>We'll do an 80-20% split for the dataset since I'm using only 60k samples, so in the train set, there's a total of 48k samples and 12k samples in the validation set.</p>
<br>
<h2>Tokenizer to convert text to Sequence.</h2>
<p>We'll initialize the tokenizer with English and Spanish Language. Two different tokenizers and convert the text to Sequence. </p>
<br>
<p>I have done some analysis on the size of the input sequence and computed the percentile. And I found the English and Spanish sequence size and did the padding with that.<p> 

<h2>Tensorflow Dataset
This is important to</h2>
<p>create the Tensorflow dataset for faster computation. So I built the TensorFlow dataset on a train and validated both. I also defined the batch size.</p>

<h2>Encoder</h2>
<p>I defined the Encoder thorough model subclassing. Here Encoder model has Embedding and GRU layer. Also, the initialize_hidden_state method is there to initialize the Encoder GRU. </p>

<p>So the input to the Encoder instance will be (English input sequence, initial_state). Input sequence will be given in batches. </p>

<p>The output from the Encoder is going to be - (work from hidden states) and (; last hidden state output)<p>

<br>
<h2>Decoder</h2>
<p>Input to Decoder is  (spanish input sequence, encoder_hidden_output, encoder_last_output)</p>

<p>The output from Decoder would be the predicted labels.</P> 
<br>

<h2>Attention Layer</h2>
<p>The attention layer would compute the class weight on each input given to the encoder side and context vector. </p>

<p>This is how the attention layer works:</p>
<ul>
  <li>First, we'll compute the score on each input of the Encoder. There’s different way to compute the score but we’re doing this wary : score = FC(tanh(FC(Encoder output) + FC(Hidden state of priv step)))</li>
  <li>Then we'll apply softmax on the score to get attention weight, i.e., attention weight =  softmax(score).  Now we weight each encoder input.</li>
  <li>From attention weight we’ll compute context vector i.e, sum(attention weights * Encoder Output)</li>
  <li>Now this attention weight and context vector are returned from the attention layer. </li>
  
 </ul>





