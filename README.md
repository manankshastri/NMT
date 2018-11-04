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
