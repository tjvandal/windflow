from . import warper, FlowNetS, PWCNet, RAFT, FlowNet2, MaskFlownet

def get_flow_model(model_name, small=True):
    # select base model
    model_name = model_name.lower()
    if model_name == 'pwc-net':
        model = PWCNet.PWCDCNet()
    elif model_name == 'flownets':
        model = FlowNetS.FlowNetS(input_channels=2)
    elif model_name == 'flownet2':
        model = FlowNet2.FlowNet2(input_channels=1)
    elif model_name == 'raft':
        raft_args ={'small': small,
                    'lr': 1e-5,
                    'mixed_precision': True,
                    'dropout': 0.0,
                    'corr_levels': 4,
                    'corr_radius': 4}
        model = RAFT(raft_args)
    elif model_name == 'maskflownet':
        model = MaskFlownet(in_ch=1)
    else:
        model = None

    return model
