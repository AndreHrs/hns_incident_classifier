from .train_loop import train_model_loop

def train_loop(model, optimiser, train_dl, valid_dl, epochs, device, patience, 
                criterion_weights, model_type='Simple', save=True):
    
     train_model_loop(model, optimiser, train_dl, valid_dl, epochs, device, patience, criterion_weights, model_type=model_type, save=save)