from .raft_trainer import RAFTTrainer, RAFTGuided
from .flownet_trainer import FlownetTrainer, MultiFrameTrainer

def get_flow_trainer(model, model_name, model_path, distribute=False, rank=0, lr=1e-4, loss='L2'):
        # Select trainer
    if model_name.lower() == 'raft':
        trainer = RAFTTrainer(model, model_path, distribute=distribute, 
                              rank=rank, lr=lr, iters=24)
    elif model_name.lower() in ['flownets', 'pwc-net', 'flownet2', 'flownetc', 'flownetsd', 'pwcnet', 'maskflownet']:
        trainer = FlownetTrainer(model, model_name, model_path,
                                 distribute=distribute, rank=rank, lr=lr,
                                 loss=loss)
        #trainer = MultiFrameTrainer(model, model_name, model_path,
        #                         distribute=distribute, rank=rank, lr=lr)
    return trainer
