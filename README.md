# biolearning

This project is intended as a quick experiment/demonstration of the training of Recurrent Spiking Neural Networks (RSNNs) as compared to Transformers.
It evaluates the performance of RSNNs on a time-series noise-rejecting classification task before training the best-performing model on a character-level NLP task in comparison to two Transformers.

Optimization tests
The RSNNs are evaluated with regards to specific changes:
1. Network size, ranging from 25-400 neurons
2. Excitation/Inhibition Ratio, from 0.5-1.00 E/(E+I)
3. Sparsity, from 0.02-1
4. Timesteps per token, from 1-8
5. Neuron Model (Integrate-and-Fire (IF), Leaky IF (LIF), Adaptive LIF (aLIF), Stochastic LIF (sLIF), Izhikevich)
6. Learning Method (Simple Reservoir Readout, Spike-Timing-Dependent Plasticity (STDP), Reward-Modulated STDP (R-STDP), Eligibility Propagation (e-Prop), Backpropagation Through Time (BPTT))
Prior to each individual test, networks ran on a sweep of learning rates to find optimal training values. Each test was ran 3 times, with reported uncertainty reflected in the graphs. Output values were test bits per character, in which a lower score is better, and percent accuracy, with an optimal value of 100%. Also recorded were the training times, in seconds, for any tests to which it is relevant.

[Results]
[Insert and format graphs.]

NLP Task
The best-performing RSNN as determined by each individual optimization test was compared to two Transformer models: one contained an equivalent number of parameters as the RSNN model, while the other one contined the same amount of *effective* parameters as the RSNN (affected by sparsisty). Each model was trained on a character-by-character level on the TinyShakespeare dataset. Training graphs recorded over 3 seeded runs were produced, as well as charts recording computation time per model.

[Results]
[Insert and format graphs.]

## Conclusions

## Full Disclosure
This project was made in collaboration with Claude Sonnet 5.5. The initial concept for this project, as well as each individual test was an original idea. The contribution of the AI in this case was the actual execution of writing the code. Each experiment was ran manually over the course of 2-3 hours on a Surface Pro, but these experiments would all be massively accelerated with GPU access. Aside from simple formatting, visual design, and the usage guide this writeup was typed entirely by hand without AI assistance. All AI-generated content was reviewed.

## Usage[]
[Describe usage]

# Appendix
## Neuron Models
### IF
### LIF
### aLIF
### sLIF
### Izhikevich

## Learning Methods
### Reservoir
### STDP
### R-STDP
### e-Prop
### BPTT

[Citations]