from .raft_trainer import RAFTTrainer, RAFTGuided
from .flownet_trainer import FlownetTrainer, MultiFrameTrainer
from .guided_trainer import PhysicsGuidedTrainer
from .unflow_trainer import UNFlowTrainer

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

def get_guided_trainer(model, model_name, model_path, lambda_obs=0.1, 
                       distribute=False, rank=0, lr=1e-4, device=None):
    if model_name == 'raft':
        trainer = RAFTGuided(model, model_path, lambda_obs=0.1, lr=lr,
                             distribute=distribute, rank=rank)
    elif model_name.lower() in ['flownets', 'pwc-net', 'pwcnet', 'maskflownet']:
        trainer = PhysicsGuidedTrainer(model_path=model_path, 
                                       model=model, levels=5, device=device, 
                                       lambda_obs=lambda_obs, lr=lr)
    return trainer
