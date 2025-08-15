########################################################################################################################
# Add controlnet options if have controlnet_aux installed (https://github.com/Fannovel16/comfyui_controlnet_aux)
# Default widgets: boolean toggle always available; preprocessor dropdown is a placeholder unless the add-on is present
use_controlnet_widget = ("BOOLEAN", {"default": False})
preprocessor_widget = (["_"],)

printout = "Attempting to add Control Net options to the 'HiRes-Fix Script' Node (comfyui_controlnet_aux add-on)..."
try:
    with suppress_output():
        AIO_Preprocessor = getattr(import_module("comfyui_controlnet_aux.__init__"), 'AIO_Preprocessor')
    preprocessor_widget = AIO_Preprocessor.INPUT_TYPES()["optional"]["preprocessor"]
    print(f"\r{message('Efficiency Nodes:')} {printout}{success('Success!')}")
except Exception:
    print(f"\r{message('Efficiency Nodes:')} {printout}{error('Failed!')}")