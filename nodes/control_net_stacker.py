# TSC Control Net Stacker
class TSC_Control_Net_Stacker:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"control_net": ("CONTROL_NET",),
                             "image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})},
                "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
                }

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("CNET_STACK",)
    FUNCTION = "control_net_stacker"
    CATEGORY = "Efficiency Nodes/Stackers"

    def control_net_stacker(self, control_net, image, strength, start_percent, end_percent, cnet_stack=None):
        # If control_net_stack is None, initialize as an empty list
        cnet_stack = [] if cnet_stack is None else cnet_stack

        # Extend the control_net_stack with the new tuple
        cnet_stack.extend([(control_net, image, strength, start_percent, end_percent)])

        return (cnet_stack,)
