import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import torch  
import torch.nn as nn                    # PyTorch
from sklearn.model_selection import train_test_split
from universal import logger
"""
    62(in) to 50(h1) to 40(h2) to 30(h3) to output(27)

"""

# Flask server
app = Flask(__name__)
CORS(app)                  # allows other browsers to connect.

epochs = 500
alphabet_dict = {letter: idx for idx, letter in enumerate('abcdefghijklmnopqrstuvwxyz')}        # LEARN IT

reverse_alphabet_dict = {idx: letter for idx, letter in enumerate('abcdefghijklmnopqrstuvwxyz')}

data = pd.read_csv("training_landmarks.csv",header = 0)          # No backslash
test,train = train_test_split(data,test_size=0.2,train_size=0.80,shuffle=True)
test_output, test_input,train_output, train_input = test.iloc[:,0].to_numpy(), test.iloc[:,1:].to_numpy(), train.iloc[:,0].to_numpy(), train.iloc[:,1:].to_numpy()

test_output, train_output = np.vectorize(alphabet_dict.get)(test_output), np.vectorize(alphabet_dict.get)(train_output)   # .get makes it callable, vectorise turns every element in array with funcion/dict
test_input, train_input = test_input.astype(np.float32), train_input.astype(np.float32)             # .iloc[] splits class, to_numpy() turns into numpy array
                                                                                                    # test.iloc[:2] split rows, test.iloc[:,2] splits columns
# Changes into tensor
test_output, test_input,train_output, train_input = torch.from_numpy(test_output), torch.from_numpy(test_input), torch.from_numpy(train_output), torch.from_numpy(train_input)      
logger("test output dtype", {test_output.dtype})        # cross entropy needs int64
logger("test input dtype", {test_input.dtype})

class neural_network(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()                                # initialises nn.module, to use it.
        self.layers = nn.Sequential(
            # input to hidden 1
            nn.Linear(input_size,50),
            nn.ReLU(),
            # hidden 1 to hidden 2
            nn.Linear(50,40),
            nn.ReLU(),
            # hidden 2 to hidden 3
            nn.Linear(40,30),
            nn.ReLU(),
            # hidden 3 to output
            nn.Linear(30,output_size)
        )
    def forward(self, input_matrix):
        z_out = self.layers(input_matrix)
        return z_out
    
    def print_weights_size(self):
        logger("input to hidden1",{self.layers[0].weight.shape},debug=False)        # EVEN indicies are the ReLU's
        logger("hidden1 to hidden2",{self.layers[2].weight.shape},debug=False)
        logger("hidden2 to hidden3",{self.layers[4].weight.shape},debug=False)
        logger("hidden3 to output",{self.layers[6].weight.shape},debug=False,gap=True)

    def print_weights_max(self):
        logger("Max weight input", self.layers[0].weight.max(), debug=False)    # CAREFUL can't use numpy array functions on tensors, use tensor method
        logger("Max weight hidden 1", self.layers[2].weight.max(), debug=False)
        logger("Max weight hidden 2", self.layers[4].weight.max(), debug=False)
        logger("Max weight output", self.layers[6].weight.max(), debug=False, gap=True)

    def print_gradient(self):
        logger("gradient input",self.layers[0].weight.grad.max(), debug= False)
        logger("gradient hidden 1",self.layers[2].weight.grad.max(), debug= False)
        logger("gradient hidden 2",self.layers[4].weight.grad.max(), debug= False)
        logger("gradient output",self.layers[6].weight.grad.max(), debug= False, gap=True)
    
    def print_loss(self,loss):
        logger("loss",loss.item(),debug=True)

def run_model (input_matrix, output_matrix):     
    neural_network.print_weights_size(model)
    optimizer.zero_grad()               # Resets gradients not weights
    z_out = model(input_matrix)
    loss = loss_fn(z_out, output_matrix)
    loss.backward()                     # backward prop
    optimizer.step()                    # updates weights
    model.print_loss(loss)

# Creates model from class, sets up optimiser, chooses loss function
model = neural_network(63,26)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
# Training
for i in range(epochs):
    run_model(train_input,train_output)

# Accuracy test
#for i in range(test_input.shape[0]):
    #test(test_input,test_output)

# Flask receiver

@app.route('/', methods = ['POST'])     # FREEZES CODE
def receiver():                         # Don't call us, we'll call you
    data = request.get_json()
    data = torch.from_numpy(np.array(data).astype(np.float32))

    logger("sent_data",str(data.type()),debug=False)    #datatype: torch.DoubleTensor
    response = model(data).detach().numpy()
    prediction = reverse_alphabet_dict[np.argmax(response)]
    logger("prediction",prediction,debug=False)
    return jsonify({"response": prediction})

if __name__ == '__main__': 
        app.run(debug=True,host ='0.0.0.0')
