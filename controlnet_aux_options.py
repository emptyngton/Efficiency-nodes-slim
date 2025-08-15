########################################################################################################################
# Add controlnet options if have controlnet_aux installed (https://github.com/Fannovel16/comfyui_controlnet_aux)
use_controlnet_widget = preprocessor_widget = (["_"],)
if os.path.exists(os.path.join(custom_nodes_dir, "comfyui_controlnet_aux")):
    printout = "Attempting to add Control Net options to the 'HiRes-Fix Script' Node (comfyui_controlnet_aux add-on)..."
    #print(f"{message('Efficiency Nodes:')} {printout}", end="", flush=True)

    try:
        with suppress_output():
            AIO_Preprocessor = getattr(import_module("comfyui_controlnet_aux.__init__"), 'AIO_Preprocessor')
        use_controlnet_widget = ("BOOLEAN", {"default": False})
        preprocessor_widget = AIO_Preprocessor.INPUT_TYPES()["optional"]["preprocessor"]
        print(f"\r{message('Efficiency Nodes:')} {printout}{success('Success!')}")
    except Exception:
        print(f"\r{message('Efficiency Nodes:')} {printout}{error('Failed!')}")