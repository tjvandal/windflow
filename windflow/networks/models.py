def get_flow_model(model_name, small=True, scheduler_total_steps=500000):
    # select base model
    model_name = model_name.lower()
    if model_name == "pwc-net":
        from . import PWCNet
        model = PWCNet.PWCDCNet()
    elif model_name == "flownets":
        from . import FlowNetS
        model = FlowNetS.FlowNetS(input_channels=2)
    elif model_name == "flownet2":
        from . import FlowNet2
        model = FlowNet2.FlowNet2(input_channels=1)
    elif model_name == "raft":
        from . import RAFT
        model = RAFT(small=small, iters=12, log_step=100, scheduler_total_steps=scheduler_total_steps)
    elif model_name == "sea_raft":
        from . import SEARAFT
        model = SEARAFT(log_step=100, scheduler_total_steps=scheduler_total_steps)
    elif model_name == "waft":
        from . import WAFT
        model = WAFT(log_step=100, scheduler_total_steps=scheduler_total_steps)
    elif model_name == "maskflownet":
        from . import MaskFlownet
        model = MaskFlownet(in_ch=1)
    else:
        model = None

    return model
