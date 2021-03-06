    
**Receptive Field Calculation:**

Here are the formulae for calculating Receptive filed:
*	jn = jn-1 * s

j0 = 1	(initialized to zero)

where

jn is jump of output

jn-1 is jump of input

s is strides

*	rn = rn-1 +(k-1) * jn-1

r0=1	(initialized to zero)

where

rn is receptive field of output

jn-1 is receptive field of input

k is kernel size

Based on these formulate, the receptive file calculation for the following network is given in the table below

![Network](https://github.com/rednas/EVA/blob/master/session%207/Network.jpg)



Table with Receptive field calculations at each layer:

![Receptive Field Calculations](https://github.com/rednas/EVA/blob/master/session%207/Receptive%20Field%20Calculation.jpg)

Note:

In the Inception blocks, the maximum kernel size is 5x5 and hence 5x5 is used in receptive field calculation.

