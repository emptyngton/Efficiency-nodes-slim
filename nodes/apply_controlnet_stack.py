# TSC Apply ControlNet Stack
class TSC_Apply_ControlNet_Stack:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",),
                             "negative": ("CONDITIONING",)},
                "optional": {"cnet_stack": ("CONTROL_NET_STACK",)}
                }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING+","CONDITIONING-",)
    FUNCTION = "apply_cnet_stack"
    CATEGORY = "Efficiency Nodes/Stackers"

    def apply_cnet_stack(self, positive, negative, cnet_stack=None):
        if cnet_stack is None:
            return (positive, negative)

        for control_net_tuple in cnet_stack:
            control_net, image, strength, start_percent, end_percent = control_net_tuple
            controlnet_conditioning = ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image,
                                                                                 strength, start_percent, end_percent)
            positive, negative = controlnet_conditioning[0], controlnet_conditioning[1]

        return (positive, negative, )