import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
import pickle
import pandas as pd



def compute_metric(predictions, labels, usage="LogisticRegression", metric="accuracy", error_function="percept"):
    #---------------------------------------------------------------
    # Colors code to display metrics
    #---------------------------------------------------------------
    red = "\n\033[31m"
    green = "\n\033[32m"
    yellow = "\n\033[33m"
    blue = "\n\033[34m"
    magenta = "\n\033[35m"
    cyan = "\n\033[36m"
    reset = "\033[0m\n"

    if usage in ['LogisticRegression', 'MultiClassification', 'SimpleCategorisation']:
        
        size = len(labels)
        
        good_prediction= lambda prediction, label, error_function: prediction*label >=0 if error_function == "percept" \
            else prediction==label if error_function == "simple" else 5
        
        is_positive = lambda x: x >= 0 if usage == "SimpleCategorisation" else x == 1 if usage == "LogisticRegression" else x >= 0

        is_negative = lambda x: x < 0 if usage == "SimpleCategorisation" else x == 0 if usage == "LogisticRegression" else x < 0
 
        match metric:
            case "accuracy" | "acc":
                accuracy = 100*sum(good_prediction(predictions[i],labels[i],error_function) for i in range(size)) / size
                print(f" {red if accuracy < 50 else green}Your model accuracy is  {accuracy}% on this dataset {reset} ")
                return accuracy, predictions
                

            case "confusion_matrix" | "conf-mat" | "recall" | "precision" | "f1"| "au_roc":
                true_negatives, true_positives, false_negatives, false_positives = 0,0,0,0
            
                true_negatives = 0
                indexes = list(range(size))
                for i in range(size):
                    if is_negative(predictions[i]) and  good_prediction(predictions[i], labels[i],error_function):
                        true_negatives += 1
                        indexes.remove(i)
                
                tmp = indexes.copy()
                true_positives = 0
                for i in indexes:
                    if is_positive(predictions[i]) and  good_prediction(predictions[i], labels[i],error_function):
                        true_positives += 1
                        tmp.remove(i)
                
                indexes = tmp.copy()
                false_negatives = 0
                for i in tmp:
                    if is_negative(predictions[i]) and not good_prediction(predictions[i], labels[i],error_function):
                        false_negatives += 1
                        indexes.remove(i)

                false_positives = sum(is_positive(predictions[i]) and not good_prediction(predictions[i], labels[i],error_function) for i in indexes)


                if metric in ["confusion_matrix","conf-mat"]:
                    print(f" {blue}TP: {true_positives} -- TN: {true_negatives} -- FP: {false_positives} -- FN: {false_negatives} {reset} ")
                    return np.array([true_positives, true_negatives, false_positives, false_negatives]), predictions
                
                if metric in ["recall",  "f1"]:
                    recall = true_positives/(true_positives+false_positives)
                    if metric == "recall":
                        print(f"{red if recall < 0.50 else green} Recall: {recall} {reset}")
                        return recall, predictions
                
                if metric in ["precision", "f1"]:
                    precision = true_positives/(true_positives+false_negatives)
                    if metric == "precision":
                        print(f" {red if precision < 50 else green} Precision: {precision} {reset}")
                        return precision, predictions
                
                if metric == "f1":
                    f1_score = 2 / ((1/precision)+(1/recall))
                    print(f"{blue} f1 score: {f1_score} {reset} ")
                    return f1_score, predictions
                
                if metric == "au_roc":
                    tpr = true_positives/(true_positives+false_negatives)
                    fpr =  false_positives/(false_positives+true_negatives)
                    au_roc = tpr/fpr
                    print(f"{cyan} auc surface: {au_roc} {reset}")
                    return au_roc, predictions

    else:
        predictions = np.array(predictions)
        labels = np.array(labels)
        match metric:
            case "MSE":
                mse = np.mean((predictions-labels))**2
                print(f"{cyan} Mean Square Error is: {mse}{reset} ")
                return mse
            case "MSA":
                msa = np.mean((predictions-labels))
                print(f"{cyan}Mean Average Error is {msa}{reset}")
                return msa


class Dense:
    # -------------------------------------------------------------------#
    # Fully connected layer
    # -------------------------------------------------------------------#
    def __init__(self, nodes_count:int, activation_function = "reLu", is_output=False, categories=[]) -> None:
        # For the moment input should be like (x, )
        # a node is an array that contains 5 values: 
        # an id --
        # x::input data-- 
        # dE/dx:: derivative of error per x -- 
        # y:: output data = result of activation function f(x) -- 
        # dE/dy:: derivative of error per y
        # input_nodes:: the nodes of the previous layer that are connected to this node
        # CONVENTION the last node of a layer is the bias node
        self.activation = activation_function
        self.size = nodes_count +1 if not is_output else nodes_count
        self.is_output = is_output
        categories = categories[:nodes_count] if len(categories) >= nodes_count else categories
        self.nodes = [
            {'id':0, 
            'label':f'{categories[i]}' if i < len(categories) else '', 
            'x':0.0, 
            'dE/dx':0.0, 
            'y':0.0, 
            'dE/dy':0.0, 
            'input_nodes':[], 
            'output_nodes':[]
            } 
        for i in range(self.size)]
        
    def activate(self):
        match self.activation:
            case "identity" | "reLu" | "sigmoid":
                if self.activation == "identity":
                    f = lambda x: x  
                elif self.activation == "reLu":
                    f = lambda x: x if x > 0 else 0
                else: 
                    f = lambda x: math.exp(x) / (1 + math.exp(x))
                
                for node in self.nodes:
                    node['y'] = f(node['x'])

            case "softmax":
                for node in self.nodes:
                    node['y'] = math.exp(node['x'])
                total = sum(node['y'] for node in self.nodes)
                assert total >= 0
                epsilon = 1e-10
                total = max(total, epsilon)
                for node in self.nodes:
                    node['y'] /= total
            
    def derivate(self):
        match self.activation:
            case "identity" | "reLu" | "sigmoid":
                if self.activation == "identity":
                    g = lambda _: 1 
                elif self.activation == "reLu":
                    g = lambda x: 0 if x <= 0 else 1
                else: 
                    g = lambda x: (math.exp(x) / (1 + math.exp(x))) * (1 - ((math.exp(x) / (1 + math.exp(x)))))

                for node in self.nodes:
                    node['dE/dx'] = node['dE/dy'] * g(node['x'])

            case "softmax":
                for node in self.nodes:
                    node['dE/dx'] = node['dE/dy']
                total = sum(math.exp(self.nodes[i]['x']) for i in range(self.size))
                assert total >= 0
                epsilon = 1e-10
                total = max(total, epsilon)
                for i in range(self.size):
                    self.nodes[i]['dE/dx'] *= (total - math.exp(self.nodes[i]['x']))*math.exp(self.nodes[i]['x'])
                    self.nodes[i]['dE/dx'] /= total**2
                


class NetModel:    
    def __init__(self, input_shape, usage="LinearRegression") -> None:
        #-------------------------------------------------------------------
        # For the moment, I manage 1D input data
        #-------------------------------------------------------------------
        self.input_shape = input_shape
        self.layers_stack = []
        self.weights = []
        self.type = usage if usage in ['LinearRegression', 'LogisticRegression', 'MultiClassification'] else 'LinearRegression'
        self.total_nodes = 0
        self.losses = []
        self.validation_losses = []
        self.data_real_classes = []
        self.multi_one_hot_encoding = []
        # Add input layer
        self.add_layer(Dense(self.input_shape[0], "identity"))
        
    
    def add_layer(self,layer: Dense):
        self.layers_stack.append(layer)
        
    
    def compile(self, n_nodes=1, categories=[], output_function = "", initializer = "xavier"):
        """
        n_nodes:: Output layer nodes number; >2 for multiclassification

        """
        #-------------------------------------------------------------------
        # Define output layer activation function
        #-------------------------------------------------------------------
        assert output_function in ["sigmoid", "softmax", "reLu", ""]
        if self.type == 'LogisticRegression':
            activation_function = "sigmoid" if output_function == "" else output_function

        elif self.type == "MultiClassification":
            activation_function = "softmax" if output_function == "" else output_function
            assert len(categories) >= 2 and len(categories) == n_nodes
            self.data_real_classes = categories
            [self.multi_one_hot_encoding.append([0, 0, 0]) for _ in range(n_nodes)]
            i = 0
            for hot_vector in self.multi_one_hot_encoding:
                hot_vector[i] = 1
                i += 1
           
        elif self.type == "LinearRegression":
            activation_function = "reLu" if output_function == "" else output_function

        #-------------------------------------------------------------------
        # Then, add the output layer
        #-------------------------------------------------------------------
        if self.type == 'MultiClassification':
            self.add_layer(Dense(n_nodes, activation_function, is_output=True, categories=categories))
        else:
            self.add_layer(Dense(1, activation_function, is_output=True))


        layers_count = len(self.layers_stack)
        self.total_nodes = sum(self.layers_stack[i].size for i in range(layers_count)) 
        
        #-------------------------------------------------------------------
        # set nodes id
        #-------------------------------------------------------------------
        count_nodes = 0
        for i in range(layers_count):
            layer = self.layers_stack[i]
            for k in range(layer.size):
                layer.nodes[k]['id'] = count_nodes + k
            count_nodes += layer.size

        #-------------------------------------------------------------------
        # add input_nodes
        # Pay attention not to add input_nodes for bias nodes
        # bias node is considered the last node of a layer
        #-------------------------------------------------------------------
        for i in range(1, layers_count):
            layer = self.layers_stack[i]
            back_layer = self.layers_stack[i-1]
            back_nodes = [back_layer.nodes[a] for a in range(back_layer.size)]
            right_range = range(layer.size -1) if i < layers_count-1 else range(layer.size)
            for k in right_range:
                layer.nodes[k]['input_nodes'] = back_nodes

        #-------------------------------------------------------------------
        # create weights and Add forward nodes
        # Weights initialization is made with Xavier initialization
        #-------------------------------------------------------------------

        i = 0
        while(i < layers_count - 1):
            layer = self.layers_stack[i]
            forward_layer = self.layers_stack[i+1]
            
            # Inputs count
            n_inputs = layer.size
            # Hidden layer nodes count
            n_hidden =  forward_layer.size if forward_layer == self.layers_stack[-1] else forward_layer.size - 1 
            match initializer:
                case "" | "xavier":
                    W = (np.random.randn(n_inputs, n_hidden) / np.sqrt(n_inputs)).flatten().tolist()
                case _:
                    W = (np.random.randn(n_inputs, n_hidden) / np.sqrt(n_inputs)).flatten().tolist()

            right_range = range(forward_layer.size-1)if i < layers_count-2 else range(forward_layer.size)
            forward_nodes = [forward_layer.nodes[i] for i in right_range ]

            i += 1
            weight_list_cursor = 0
            for k in range(layer.size):
                layer.nodes[k]['output_nodes'] = forward_nodes                
                for j in right_range:
                    new_weight = {f"w{layer.nodes[k]['id']}{forward_layer.nodes[j]['id']}": W[weight_list_cursor]}
                    self.weights.append(new_weight)
                    weight_list_cursor += 1
                
            
        # reformat weights
        self.weights = np.array(self.weights)


    def train(self,train_X, train_Y, cost_function="l2", nepochs=100, learning_rate=0.01, validation_per=0.25):
        # -------------------------------------------------------------------#
        # Transform labels into binary values if LogisticRegression Model
        # -------------------------------------------------------------------#
        train_Y_cpy = train_Y.copy()
        if self.type == 'LogisticRegression':
            train_Y_cpy = np.array(train_Y_cpy)
            self.data_real_classes = np.unique(train_Y_cpy)
            assert len(self.data_real_classes) == 2
            train_Y_cpy = np.where((train_Y_cpy == min(self.data_real_classes)), 0, 1)

        if self.type == "MultiClassification":
            for i in range(len(train_Y_cpy)):
                index = self.data_real_classes.index(train_Y_cpy[i]) 
                train_Y_cpy[i] = self.multi_one_hot_encoding[index]
        
        size = int(len(train_Y_cpy)*(validation_per))
        validation_X = train_X[0:size]
        validation_Y = train_Y_cpy[0:size]
        train_X = train_X[size:]
        train_Y_cpy = train_Y_cpy[size:]

        for _ in range(nepochs):
            # -------------------------------------------------------------------#
            # Iterate over the samples
            # -------------------------------------------------------------------#
            loss_history = np.array([])

            for n in range(len(train_X)):
                # -------------------------------------------------------------------#
                # compute output
                # -------------------------------------------------------------------#
                self.forward_propagation(train_X[n])
        
                # -------------------------------------------------------------------#
                # Compute error
                # -------------------------------------------------------------------#
                if self.type != "MultiClassification":
                    output_node = self.layers_stack[-1].nodes[0]
                    E, output_node['dE/dy']= self.compute_error(output_node['y'], train_Y_cpy[n], cost_function)
                    loss_history = np.append(loss_history, E)
                else:
                    Y = [output_node['y'] for output_node in self.layers_stack[-1].nodes]
                    E, dE_per_y= self.compute_error(Y, train_Y_cpy[n], cost_function)
                    for i in range(self.layers_stack[-1].size):
                        self.layers_stack[-1].nodes[i]['dE/dy'] = dE_per_y[i]
                    loss_history = np.append(loss_history, E)
                    
                # -------------------------------------------------------------------#
                # Backpropagate
                # -------------------------------------------------------------------#
                self.backpropagate(learning_rate)
            
            this_epoch_validation_losses_history = []
            for count in range(size):
                validation_result = self.predict_unique(x = validation_X[count], display_message=False, rt_real_prediction=False)
                valid_error = self.compute_error(y_predicted=validation_result, y_recorded=validation_Y[count], cost_function=cost_function)[0]
                this_epoch_validation_losses_history.append(valid_error)
            self.validation_losses.append(np.array(this_epoch_validation_losses_history).mean())
    
            print(f"Epoch {_}:  Loss is {np.mean(loss_history)} ")
            self.losses.append(np.mean(loss_history))
        return self.weights
    

    def forward_propagation(self, input_data):
        input_layer = self.layers_stack[0]
        # -------------------------------------------------------------------#
        # Insert data into the input layer
        # -------------------------------------------------------------------#
        for i in range(input_layer.size - 1): 
            input_layer.nodes[i]['x'] = input_data[i]
        # Set bias node input to 1
        input_layer.nodes[-1]['x'] = 1
        input_layer.activate()
    
        # -------------------------------------------------------------------#
        # forward propagation
        # -------------------------------------------------------------------#
        for i in range(1, len(self.layers_stack)):
            layer = self.layers_stack[i]
            for j in range(layer.size):
                # retrieve all nodes linked to this one in the previous layer
                if layer.nodes[j]['input_nodes']:
                    involved_weights = [self.get_weight_value(self.get_weight(str(f"w{n_node['id']}{layer.nodes[j]['id']}"))) 
                                        for n_node in layer.nodes[j]['input_nodes']]
                    x_i = [n_node['y'] for n_node in layer.nodes[j]['input_nodes']]
                    
                    x_i = np.array(x_i).astype('float64')
                    involved_weights = np.array(involved_weights).astype('float64')
                    layer.nodes[j]['x'] = involved_weights.T.dot(x_i)
                else:
                    # Bias nodes probably
                    layer.nodes[j]['x'] = 1

            layer.activate()
        
        output_layer = self.layers_stack[-1]

        if self.type != 'MultiClassification': return output_layer.nodes[0]['y']
        else: return [output_layer.nodes[i]['y'] for i in range(output_layer.size)]
    

    def compute_error(self, y_predicted, y_recorded, cost_function="l2"):
        match cost_function:
            
            case "l2" | "quadratic":
                error = lambda y, y_prim: (y_prim-y)**2
                derivative = lambda y, y_prim: 2*(y_prim-y)
            
            case "l1":
                error = lambda y, y_prim: abs(y-y_prim)
                derivative = lambda _, o: 1
    
            case "cross_entropy":
                epsilon = 1e-10
                error = lambda y, y_prim: -y*math.log(max(min(y_prim, 1-epsilon), epsilon)) - (1-y)*math.log(max(min(1-y_prim, 1-epsilon), epsilon))
                derivative = lambda y, y_prim: (max(min(y_prim, 1-epsilon), epsilon)-y)/(max(min(y_prim, 1-epsilon), epsilon)*(max(min(1-y_prim, 1-epsilon), epsilon)))
                
        if type(y_predicted) not in [np.array, list] :
            return error(y_recorded, y_predicted), derivative(y_recorded, y_predicted)
        else:
            return (sum(error(y_recorded[i], y_predicted[i]) for i in range(len(y_recorded))), [derivative(y_recorded[i], y_predicted[i]) for i in range(len(y_recorded))] )
            

    def backpropagate(self, learning_rate):
        # -------------------------------------------------------------------#
        # Backpropagate the prediction error into all hidden layers
        # -------------------------------------------------------------------#
        for i in range(len(self.layers_stack)-1, -1, -1):
            layer =  self.layers_stack[i]
            p_layer = self.layers_stack[i-1] if i-1 >= 0 else None
            
            # For each node, update dE/dx
            layer.derivate()
                                  
            for layer_node in layer.nodes:
                # -------------------------------------------------------------------#
                # update weights linked to this node
                # -------------------------------------------------------------------#
                if p_layer:
                    for linked_node, id in [(l, str(l['id'])) for l in layer_node['input_nodes']]:
                        w = self.get_weight(f"w{id}{layer_node['id']}")
                        self.update_weight(w, -learning_rate*layer_node['dE/dx']*linked_node['y'])
                        
            # -------------------------------------------------------------------#
            # update dE/dy for layer before
            # -------------------------------------------------------------------#
            if p_layer:
                for linked_node in p_layer.nodes:
                    weight_i_j = np.array([self.get_weight_value(self.get_weight(f"w{linked_node['id']}{node_['id']}")) for node_ in linked_node['output_nodes']])
                    derivatives_per_x_j = np.array([node_['dE/dx'] for node_ in linked_node['output_nodes']])
                    linked_node['dE/dy'] = derivatives_per_x_j.T.dot(weight_i_j)

    
    def predict_unique(self, x, y=None, display_message=True, rt_real_prediction=True):
        prediction = self.forward_propagation(x)
        match self.type:
            case "LinearRegression":
                if display_message: print(f"Prediction = {prediction} -- Target = {y}")
                return prediction

            case "LogisticRegression":
                if prediction >= 0.5:
                    r_prediction = self.data_real_classes[1]
                else:
                   r_prediction = self.data_real_classes[0]
                
                if display_message: print(f"Prediction = {r_prediction} -- Target = {y}") 
                return r_prediction if rt_real_prediction else prediction

            case "MultiClassification":
                # Probabilities distribution for each node (category)
                # Choose the most probable
                index = prediction.index(max(prediction))
                r_prediction = self.data_real_classes[index]
                #[print(f"{self.data_real_classes[i]}: {100*prediction[i]}%") for i in range(len(self.data_real_classes))]
                if display_message: print(f"Prediction = {r_prediction} -- Target = {y}") 
                return r_prediction if rt_real_prediction else prediction
                

    def predict_sample(self, X: list or np.array, Y :list or np.array, metric=""):
        assert len(X) == len(Y), "The target data are not the same size as the samples"
        # -------------------------------------------------------------------#
        # choose the appropriate metric if none is given
        # -------------------------------------------------------------------#
        if metric == "": metric = "accuracy" if self.type == "LogisticRegression" else "MSE" if self.type == "LinerRegression" else "accuracy"      
        predictions = []
        for record in X:
            predictions.append(self.predict_unique(record, display_message=False))
        
        if self.type == 'LogisticRegression':
            _Y = np.array(Y)
            _Y = np.where((_Y == min(self.data_real_classes)), 0, 1)
            _predictions = np.array(predictions)
            _predictions = np.where((_predictions == min(self.data_real_classes)), 0, 1)
            compute_metric(predictions=_predictions, labels=_Y, usage=self.type, metric=metric, error_function="simple")

        else: 
            compute_metric(predictions=predictions, labels=Y, usage=self.type, metric=metric, error_function="simple")
        return predictions
    

    def get_weight(self, label):
        for w in self.weights:
            if label in w:
                return w


    def get_weight_value(self, weight):
        return next(iter(weight.values()))


    def update_weight(self, weight, value):
        weight[list(weight.keys())[0]] += value
        pass

                            
    def describe(self):
        # function to describe the layers and the nodes
        i = 0
        for layer in self.layers_stack:
            print(f"Layer {str(i)}")
            i += 1
            tmp = []
            for _node in layer.nodes:
                tmp.append((_node['id'], _node['label'], len(_node['input_nodes']), len(_node['output_nodes'])))
                
            tmp = pd.DataFrame(tmp, columns=['id', 'label', 'inputs', 'outputs']).set_index('id')
            print(tmp)


    def draw(self):
        fig, ax = plt.subplots()
        y = 0
        r = 0.5
        offset = 4
        x = r
        visual_nodes = [0]*self.total_nodes
        beautiful_colors = [(0.25,0.88,0.82),
                            (0.8,0.6,1),
                            (1,0.7,0.5),
                            (1,0.85,0.35),
                            (0.53,0.81,0.92),
                            (0.58,0.44,0.86),
                            (0.5,0.5,0),
                            (0.99,0.81,0.87)]

        for i in range(len(self.layers_stack)):
            # Draw the layer nodes
            layer = self.layers_stack[i]
            y = r
            #color = random.choice(beautiful_colors)
            color=(random.random(), random.random(),
                   random.random())
            bias = layer.nodes[-1]
            for node in layer.nodes:
                # Draw circles (nodes)                
                tmp = visual_nodes[node['id']] = Circle((x, y), r, fill=False, color=color )                
                ax.add_artist(tmp)
                ax.text(x=tmp.center[0], y=tmp.center[1], s=node['id'] if node != bias or i == len(self.layers_stack)-1 
                        else f"{node['id']} (bias)", color=color)
                y += r+1
            
            x += offset

        # Draw forward arrows
        for layer in self.layers_stack[:-1]:
            for node in layer.nodes:
                for connected_node in node['output_nodes']:
                    A = visual_nodes[node['id']].center[0]+r, visual_nodes[node['id']].center[1]
                    B = visual_nodes[connected_node['id']].center[0]-r, visual_nodes[connected_node['id']].center[1]

                    opposed_side_length = A[1] - B[1]
                    hypotenuse_length = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
                    assert hypotenuse_length >= 0
                    epsilon = 1e-10
                    hypotenuse_length = max(hypotenuse_length, epsilon)

                    angle = math.degrees(math.asin(opposed_side_length/hypotenuse_length))
                    ax.add_artist(ConnectionPatch(A, B, "data", "data", arrowstyle='->', color='blue'))
                    self.weights
                    label= self.get_weight_value(self.get_weight(f"w{node['id']}{connected_node['id']}")) 
                    ax.text(x = (A[0]+B[0])/2, y=(A[1]+B[1])/2, 
                            s=f"{label:.2f} ", rotation=-angle, 
                            rotation_mode='anchor', ha='center', va="center")
                            
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        plt.title("Model")
        fig.tight_layout()
        plt.show()
        pass


    def summarize(self):
        print(f"Input shape: {self.input_shape}\n Weights: {self.weights}\n Number of nodes: {self.total_nodes}")
    

    def save(self, model_name):
        # Save the model in the working directory
        with open(f"{model_name}.ssj","wb") as save_file:
            pickle.dump(self, save_file)
        

    def display_losses(self):
        n = len(self.losses)
        if n < 2:
            self.losses.append(self.losses[0])
            self.validation_losses.append(self.validation_losses[0])
            n += 1
        
        nepochs = range(1, n+1)

        axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))[1]
        axs[0].plot(nepochs, self.losses, "blue")
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Training Loss')
        
        axs[1].plot(nepochs, self.validation_losses, "green")
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Validation Loss')

        plt.show()
        pass


def load(model_name):
    with open(f"{model_name}.ssj","rb") as save_file:
        load = pickle.load(save_file)
        return load


if __name__ == '__main__':
    a = NetModel(input_shape=(4, ), usage="LinearRegression")
    a.add_layer(Dense(2, activation_function="sigmoid"))
    a.compile()
    a.draw()