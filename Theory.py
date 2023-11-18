'''
ACTIVATON FUNCTIONS:
    -argmax cannot be used for back propogation

FINAL FUNCTION:
    -cross entropy will take us toward a better prediction because slope 
     of cross entropy is large.

CNN:
    -takes advantage of correlation between two search boxes* which are closer to each other in image,

    -filter (aka kernel, convolution, initially some random values), compute dot product between input(r1) and kernel(r2) = r3, 
    add bias to result r3 + bias = r4, put this result into feature map(r5), after this move kernel by some pixels(stride).

    -pass feature map(r5) to activation function, apply filter(which does pooling, moves in such a way that 
     it does not overlap itself) to feature map = r6,
     
    -now use r6 in normal neural network.
    
Recurrent NN:
    -different amount of input value, has feedback loops(takes past into consideration), weights and bias are shared 
     across every input.
     
    -more we unroll(making copy of RNN, ouput of one is fed as input to other) the harder is to train. since each time we 
    copy RNN for each copy the input is multiplied by (weight), (weight)^2, (weight)^3,...(weight)^(# of unroll). so it explodes. 
    so if weight < 1, gradient vanishes, else values explodes.

    -long-short term memory(LSTM):
        --uses sigmoid activation function.

        --in first stage STM is multiplied to input + after multiplying input with weights + adding bias,(till this r1) 
          and passing through sigmoid (r2), result(r2 = [0, 1]) is multiplied with long term memory(r3), so result in 
          first stage gives what percentage of LTM is to be remembered. LTM updates to r3.
         
        --in middle result(r8) = LTM(r3) + (% potential memory to remember(sigmoid)) * (potential long term memory(tanh))
            ---% potential memory to remember: same as first stage up till r1(aka r4), r4 is passed through sigmoid r5
            ---potential long term memory: same as first stage up till r1(aka r6), r6 is passed through tanh r7
          LTM updates to r8.

        --in last stage r9(derived same as r1), r10 = r9 * (tanh(LTM)), short term memory updates to R10
        
        --*note: here same as some other result mean same block as that of the other but may vary in weight and other 
                 parameter values.

        --input(LTM, STM, input1) --LSTM model--> input(updated(LTM, STM), input2), unroll --LSTM model--> .. ans so on

BATCH NORMALIZATION:
    -standardization(mean=0, variance=(0,1)) --> scale and add offset
    -BN(x) = scale * ((x - mean) / std.deviation) + offset

DEGRADATION PROBLEM:
    -ex, suppose there are x + y layers, input of x is identical to input of y this is the problem with shallower networks.

GAN:
    -two models(generator and discriminator) compete with each other and learn patterns in input data
    -generator creates fake data to be trained on the discriminator
    -discriminator decides whether the data is from real or not with probabilitites.
'''