# Neural Machine Translation

Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25") using one of the most sophisticated sequence to sequence models - attention model.


## Attention Mechanism

The diagram on the left shows the attention model and the one on the right shows what one "Attention" step does to calculate the attention variables.

<table>
<td> 
<img src="images/attn_model.png" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="images/attn_mechanism.png" style="width:500;height:500px;"> <br>
</td> 
</table>



There are two functions: `one_step_attention()` and `model()`.

-`one_step_attention()`: At step <i>t</i>, given all the hidden states of the Bi-LSTM and the previous hidden state of the second LSTM
`one_step_attention()` will compute the attention weights and output the context vector.

-`model()`: Implements the entire model.  It first runs the input through a Bi-LSTM.Then,  it calls `one_step_attention()`  <i>Ty</i> times. At each iteration of this loop,  it gives the computed context vector to the second LSTM,  and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction. 


## Visualising Attention

Since the problem has a fixed output length of 10, it is also possible to carry out this task using 10 different softmax units to generate the 10 characters of the output. But one advantage of the attention model is that each part of the output (say the month) knows it needs to depend only on a small part of the input (the characters in the input giving the month). We can visualize what part of the output is looking at what part of the input.

Consider the task of translating "Saturday 9 May 2018" to "2018-05-09". If we visualize the computed attention we get this:

<img src="images/date_attention.png" style="width:600;height:300px;"> <br>

Notice how the output ignores the "Saturday" portion of the input. None of the output timesteps are paying much attention to that portion of the input. We also see that '9' has been translated as '09' and 'May' has been correctly translated into '05', with the output paying attention to the parts of the input it needs to make the translation. The 'year' mostly pays attention to the input's '18' in order to generate "2018."
